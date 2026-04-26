from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import ChunkEntityLink, GraphEntity, GraphRelation
from app.services.embeddings import embed_texts
from app.services.graph_extraction import ExtractedEntity, ExtractedRelation


def upsert_graph_entity(
    db: Session,
    dataset_id: int,
    entity: ExtractedEntity,
) -> GraphEntity:
    canonical = entity.name.strip().lower()
    stmt: Select = select(GraphEntity).filter(
        GraphEntity.dataset_id == dataset_id,
        GraphEntity.name.ilike(entity.name.strip()),
    )
    existing = db.execute(stmt).scalars().first()
    if existing is not None:
        return existing

    embedding = embed_texts([entity.name])[0]
    row = GraphEntity(
        dataset_id=dataset_id,
        name=entity.name.strip(),
        entity_type=entity.entity_type,
        description=entity.description,
        aliases={"canonical": canonical, "aliases": [entity.name.strip()]},
        embedding=embedding,
        entity_metadata={"source": "heuristic_extractor"},
    )
    db.add(row)
    db.flush()
    return row


def create_graph_relation(
    db: Session,
    dataset_id: int,
    relation: ExtractedRelation,
    source_entity_id: int,
    target_entity_id: int,
    evidence_chunk_id: int,
) -> GraphRelation | None:
    if source_entity_id == target_entity_id:
        return None
    stmt: Select = select(GraphRelation).filter(
        GraphRelation.dataset_id == dataset_id,
        GraphRelation.source_entity_id == source_entity_id,
        GraphRelation.target_entity_id == target_entity_id,
        GraphRelation.relation == relation.relation,
        GraphRelation.evidence_chunk_id == evidence_chunk_id,
    )
    existing = db.execute(stmt).scalars().first()
    if existing is not None:
        return existing

    row = GraphRelation(
        dataset_id=dataset_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation=relation.relation,
        weight=relation.confidence,
        evidence_chunk_id=evidence_chunk_id,
        relation_metadata={"source": "heuristic_extractor"},
    )
    db.add(row)
    db.flush()
    return row


def create_chunk_entity_link(
    db: Session,
    chunk_id: int,
    entity_id: int,
    confidence: float,
) -> ChunkEntityLink:
    stmt: Select = select(ChunkEntityLink).filter(
        ChunkEntityLink.chunk_id == chunk_id,
        ChunkEntityLink.entity_id == entity_id,
    )
    existing = db.execute(stmt).scalars().first()
    if existing is not None:
        return existing

    row = ChunkEntityLink(
        chunk_id=chunk_id,
        entity_id=entity_id,
        confidence=confidence,
        link_metadata={"source": "heuristic_extractor"},
    )
    db.add(row)
    db.flush()
    return row
