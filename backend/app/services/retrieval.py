from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import Chunk, DataFile
from app.schemas import SourceChunk
from app.services.embeddings import embed_query


def retrieve_similar_chunks(
    db: Session, query: str, k: int, dataset_id: int | None = None
) -> list[SourceChunk]:
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
