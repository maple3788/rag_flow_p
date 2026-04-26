from sqlalchemy.orm import Session

from app.config import settings
from app.models import Chunk
from app.services.graph_extraction import extract_graph_elements
from app.services.graph_store import create_chunk_entity_link, create_graph_relation, upsert_graph_entity


def build_graph_from_chunks(db: Session, dataset_id: int, chunks: list[Chunk]) -> None:
    if not settings.graphrag_enabled or not chunks:
        return

    for chunk in chunks:
        entities, relations = extract_graph_elements(chunk.content)
        if not entities:
            continue
        entity_map: dict[str, int] = {}
        for entity in entities:
            row = upsert_graph_entity(db=db, dataset_id=dataset_id, entity=entity)
            entity_map[row.name.strip().lower()] = int(row.id)
            create_chunk_entity_link(
                db=db,
                chunk_id=int(chunk.id),
                entity_id=int(row.id),
                confidence=entity.confidence,
            )

        for relation in relations:
            source_id = entity_map.get(relation.source_name.strip().lower())
            target_id = entity_map.get(relation.target_name.strip().lower())
            if source_id is None or target_id is None:
                continue
            create_graph_relation(
                db=db,
                dataset_id=dataset_id,
                relation=relation,
                source_entity_id=source_id,
                target_entity_id=target_id,
                evidence_chunk_id=int(chunk.id),
            )
