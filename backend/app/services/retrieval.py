from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import Chunk, DataFile
from app.schemas import SourceChunk
from app.services.bm25 import bm25_search
from app.services.embeddings import embed_query


def retrieve_similar_chunks(
    db: Session, query: str, k: int, dataset_id: int | None = None
) -> list[SourceChunk]:
    if dataset_id is not None:
        bm25_hits = bm25_search(query=query, dataset_id=dataset_id, top_k=k)
        if bm25_hits:
            chunk_ids = [hit.chunk_id for hit in bm25_hits]
            score_by_chunk_id = {hit.chunk_id: hit.score for hit in bm25_hits}
            stmt: Select = (
                select(Chunk, DataFile)
                .join(DataFile, Chunk.file_id == DataFile.id)
                .filter(Chunk.dataset_id == dataset_id, Chunk.id.in_(chunk_ids))
            )
            rows = db.execute(stmt).all()
            chunk_map = {chunk.id: (chunk, file_row) for chunk, file_row in rows}
            ordered_sources: list[SourceChunk] = []
            for chunk_id in chunk_ids:
                row = chunk_map.get(chunk_id)
                if row is None:
                    continue
                chunk, file_row = row
                ordered_sources.append(
                    SourceChunk(
                        dataset_id=chunk.dataset_id,
                        file_id=file_row.id,
                        filename=file_row.filename,
                        chunk_id=chunk.id,
                        content=chunk.content,
                        metadata=chunk.chunk_metadata,
                        score=score_by_chunk_id.get(chunk.id, 0.0),
                    )
                )
            if ordered_sources:
                return ordered_sources

    query_embedding = embed_query(query)
    score_expr = Chunk.embedding.cosine_distance(query_embedding).label("score")
    stmt: Select = select(Chunk, DataFile, score_expr).join(DataFile, Chunk.file_id == DataFile.id)
    if dataset_id is not None:
        stmt = stmt.filter(Chunk.dataset_id == dataset_id)
    stmt = stmt.order_by(score_expr.asc()).limit(k)
    rows = db.execute(stmt).all()

    sources: list[SourceChunk] = []
    for chunk, file_row, score in rows:
        sources.append(
            SourceChunk(
                dataset_id=chunk.dataset_id,
                file_id=file_row.id,
                filename=file_row.filename,
                chunk_id=chunk.id,
                content=chunk.content,
                metadata=chunk.chunk_metadata,
                score=float(score),
            )
        )
    return sources
