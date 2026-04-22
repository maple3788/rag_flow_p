from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import Chunk, DataFile
from app.schemas import SourceChunk
from app.config import settings
from app.services.bm25 import bm25_search
from app.services.embeddings import embed_query
from app.services.faiss_index import dense_search
from app.services.rrf import reciprocal_rank_fusion
from app.services.reranker import rerank_fused_chunks


def retrieve_similar_chunks(
    db: Session,
    query: str,
    k: int,
    dataset_id: int | None = None,
    top_k_bm25: int | None = None,
    top_k_dense: int | None = None,
    fusion_method: str = "rrf",
    rerank_enabled: bool = True,
    rerank_model: str | None = None,
) -> list[SourceChunk]:
    final_k = max(1, k)
    query_embedding = embed_query(query)

    if dataset_id is not None:
        bm25_limit = max(1, top_k_bm25 or final_k)
        dense_limit = max(1, top_k_dense or final_k)
        candidate_limit = max(final_k, bm25_limit, dense_limit)
        dense_hits = dense_search(
            query_embedding=query_embedding,
            dataset_id=dataset_id,
            top_k=dense_limit,
        )
        bm25_hits = bm25_search(query=query, dataset_id=dataset_id, top_k=bm25_limit)
        if dense_hits or bm25_hits:
            fusion_name = (fusion_method or "rrf").strip().lower()
            if fusion_name == "rrf":
                fused = reciprocal_rank_fusion(
                    ranked_lists=[
                        [hit.chunk_id for hit in bm25_hits],
                        [hit.chunk_id for hit in dense_hits],
                    ],
                    rrf_k=settings.rrf_k,
                )
            else:
                combined_ids = [hit.chunk_id for hit in bm25_hits] + [
                    hit.chunk_id for hit in dense_hits
                ]
                deduped_ids: list[int] = []
                seen: set[int] = set()
                for chunk_id in combined_ids:
                    if chunk_id in seen:
                        continue
                    seen.add(chunk_id)
                    deduped_ids.append(chunk_id)
                fused = [(chunk_id, 0.0) for chunk_id in deduped_ids]
            if fused:
                fused = fused[:candidate_limit]
                chunk_ids = [chunk_id for chunk_id, _ in fused]
                score_by_chunk_id = {chunk_id: score for chunk_id, score in fused}
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
                    if rerank_enabled:
                        return rerank_fused_chunks(
                            query=query,
                            chunks=ordered_sources,
                            top_k=final_k,
                            model_name=rerank_model,
                        )
                    return ordered_sources[:final_k]

    score_expr = Chunk.embedding.cosine_distance(query_embedding).label("score")
    stmt: Select = select(Chunk, DataFile, score_expr).join(DataFile, Chunk.file_id == DataFile.id)
    if dataset_id is not None:
        stmt = stmt.filter(Chunk.dataset_id == dataset_id)
    stmt = stmt.order_by(score_expr.asc()).limit(final_k)
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
