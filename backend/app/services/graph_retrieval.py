from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk, ChunkEntityLink, DataFile, GraphEntity, GraphRelation
from app.schemas import SourceChunk
from app.services.embeddings import embed_query


def retrieve_graph_context(
    db: Session,
    query: str,
    dataset_id: int,
    final_k: int,
    entity_top_k: int | None = None,
    relation_top_k: int | None = None,
    chunk_top_k: int | None = None,
) -> list[SourceChunk]:
    query_embedding = embed_query(query)
    top_entities = _retrieve_seed_entities(
        db=db,
        dataset_id=dataset_id,
        query_embedding=query_embedding,
        limit=entity_top_k or settings.graphrag_entity_top_k,
    )
    if not top_entities:
        return []

    entity_ids = [entity.id for entity in top_entities]
    expanded_entity_ids = _expand_neighbors(
        db=db,
        dataset_id=dataset_id,
        seed_entity_ids=entity_ids,
        relation_limit=relation_top_k or settings.graphrag_relation_top_k,
    )
    chunk_ids = _retrieve_supporting_chunks(
        db=db,
        entity_ids=expanded_entity_ids,
        limit=chunk_top_k or settings.graphrag_chunk_top_k,
    )
    if not chunk_ids:
        return []
    return _to_sources(
        db=db,
        dataset_id=dataset_id,
        chunk_ids=chunk_ids[: max(1, final_k)],
    )


def _retrieve_seed_entities(
    db: Session,
    dataset_id: int,
    query_embedding: list[float],
    limit: int,
) -> list[GraphEntity]:
    score_expr = GraphEntity.embedding.cosine_distance(query_embedding).label("score")
    stmt: Select = (
        select(GraphEntity)
        .filter(GraphEntity.dataset_id == dataset_id, GraphEntity.embedding.is_not(None))
        .order_by(score_expr.asc())
        .limit(max(1, limit))
    )
    return list(db.execute(stmt).scalars().all())


def _expand_neighbors(
    db: Session,
    dataset_id: int,
    seed_entity_ids: list[int],
    relation_limit: int,
) -> list[int]:
    if not seed_entity_ids:
        return []
    stmt: Select = (
        select(GraphRelation)
        .filter(
            GraphRelation.dataset_id == dataset_id,
            GraphRelation.source_entity_id.in_(seed_entity_ids)
            | GraphRelation.target_entity_id.in_(seed_entity_ids),
        )
        .order_by(GraphRelation.weight.desc())
        .limit(max(1, relation_limit))
    )
    relations = db.execute(stmt).scalars().all()
    entity_ids = set(seed_entity_ids)
    for relation in relations:
        entity_ids.add(int(relation.source_entity_id))
        entity_ids.add(int(relation.target_entity_id))
    return list(entity_ids)


def _retrieve_supporting_chunks(db: Session, entity_ids: list[int], limit: int) -> list[int]:
    if not entity_ids:
        return []
    stmt: Select = (
        select(ChunkEntityLink.chunk_id)
        .filter(ChunkEntityLink.entity_id.in_(entity_ids))
        .order_by(ChunkEntityLink.confidence.desc())
        .limit(max(1, limit))
    )
    rows = db.execute(stmt).all()
    deduped: list[int] = []
    seen: set[int] = set()
    for (chunk_id,) in rows:
        numeric = int(chunk_id)
        if numeric in seen:
            continue
        seen.add(numeric)
        deduped.append(numeric)
    return deduped


def _to_sources(db: Session, dataset_id: int, chunk_ids: list[int]) -> list[SourceChunk]:
    if not chunk_ids:
        return []
    stmt: Select = (
        select(Chunk, DataFile)
        .join(DataFile, Chunk.file_id == DataFile.id)
        .filter(Chunk.dataset_id == dataset_id, Chunk.id.in_(chunk_ids))
    )
    rows = db.execute(stmt).all()
    by_id = {int(chunk.id): (chunk, data_file) for chunk, data_file in rows}
    sources: list[SourceChunk] = []
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        row = by_id.get(int(chunk_id))
        if row is None:
            continue
        chunk, data_file = row
        sources.append(
            SourceChunk(
                dataset_id=int(chunk.dataset_id),
                file_id=int(data_file.id),
                filename=str(data_file.filename),
                chunk_id=int(chunk.id),
                content=str(chunk.content),
                metadata=chunk.chunk_metadata,
                score=float(1.0 / rank),
            )
        )
    return sources
