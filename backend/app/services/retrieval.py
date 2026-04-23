from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import Chunk, DataFile
from app.schemas import SourceChunk
from app.config import settings
from app.services.bm25 import bm25_search, bm25_search_file_summaries
from app.services.embeddings import embed_query
from app.services.faiss_index import dense_search, dense_search_file_summaries
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
        debug = build_dataset_retrieval_debug(
            db=db,
            query=query,
            dataset_id=dataset_id,
            final_k=final_k,
            top_k_bm25=top_k_bm25,
            top_k_dense=top_k_dense,
            fusion_method=fusion_method,
            rerank_enabled=rerank_enabled,
            rerank_model=rerank_model,
            query_embedding=query_embedding,
        )
        if debug["final_sources"]:
            return debug["final_sources"]

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


def build_dataset_retrieval_debug(
    db: Session,
    query: str,
    dataset_id: int,
    final_k: int,
    top_k_bm25: int | None,
    top_k_dense: int | None,
    fusion_method: str,
    rerank_enabled: bool,
    rerank_model: str | None,
    use_summary: bool = False,
    file_router_top_k: int = 8,
    query_embedding: list[float] | None = None,
) -> dict:
    effective_embedding = query_embedding or embed_query(query)
    candidate_file_ids: list[int] = []
    if use_summary:
        candidate_file_ids = _route_candidate_files(
            query=query,
            dataset_id=dataset_id,
            query_embedding=effective_embedding,
            top_k=max(1, file_router_top_k),
        )
    bm25_limit = max(1, top_k_bm25 or final_k)
    dense_limit = max(1, top_k_dense or final_k)
    candidate_limit = max(final_k, bm25_limit, dense_limit)

    dense_hits = dense_search(
        query_embedding=effective_embedding,
        dataset_id=dataset_id,
        top_k=dense_limit,
    )
    bm25_hits = bm25_search(query=query, dataset_id=dataset_id, top_k=bm25_limit)

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
        combined_ids = [hit.chunk_id for hit in bm25_hits] + [hit.chunk_id for hit in dense_hits]
        deduped_ids: list[int] = []
        seen: set[int] = set()
        for chunk_id in combined_ids:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            deduped_ids.append(chunk_id)
        fused = [(chunk_id, 0.0) for chunk_id in deduped_ids]

    fused = fused[:candidate_limit]
    chunk_ids = [chunk_id for chunk_id, _ in fused]
    score_by_chunk_id = {chunk_id: score for chunk_id, score in fused}
    fused_sources = _fetch_sources_by_chunk_ids(
        db=db,
        dataset_id=dataset_id,
        chunk_ids=chunk_ids,
        score_by_chunk_id=score_by_chunk_id,
        allowed_file_ids=set(candidate_file_ids) if candidate_file_ids else None,
    )
    if not fused_sources and candidate_file_ids:
        fused_sources = _fetch_sources_by_chunk_ids(
            db=db,
            dataset_id=dataset_id,
            chunk_ids=chunk_ids,
            score_by_chunk_id=score_by_chunk_id,
            allowed_file_ids=None,
        )
    reranked_sources = (
        rerank_fused_chunks(
            query=query,
            chunks=fused_sources,
            top_k=len(fused_sources),
            model_name=rerank_model,
        )
        if rerank_enabled and fused_sources
        else fused_sources
    )
    final_sources = reranked_sources[:final_k]
    return {
        "bm25_hits": bm25_hits,
        "dense_hits": dense_hits,
        "fused_hits": fused,
        "summary_used": use_summary,
        "file_hits": candidate_file_ids,
        "reranked_sources": reranked_sources,
        "final_sources": final_sources,
    }


def _fetch_sources_by_chunk_ids(
    db: Session,
    dataset_id: int,
    chunk_ids: list[int],
    score_by_chunk_id: dict[int, float],
    allowed_file_ids: set[int] | None = None,
) -> list[SourceChunk]:
    if not chunk_ids:
        return []
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
        if allowed_file_ids is not None and file_row.id not in allowed_file_ids:
            continue
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
    return ordered_sources


def _route_candidate_files(
    query: str,
    dataset_id: int,
    query_embedding: list[float],
    top_k: int,
) -> list[int]:
    bm25_hits = bm25_search_file_summaries(query=query, dataset_id=dataset_id, top_k=top_k)
    dense_hits = dense_search_file_summaries(
        query_embedding=query_embedding,
        dataset_id=dataset_id,
        top_k=top_k,
    )
    if not bm25_hits and not dense_hits:
        return []
    fused = reciprocal_rank_fusion(
        ranked_lists=[
            [hit.file_id for hit in bm25_hits],
            [hit.file_id for hit in dense_hits],
        ],
        rrf_k=settings.rrf_k,
    )
    return [file_id for file_id, _ in fused[:top_k]]
