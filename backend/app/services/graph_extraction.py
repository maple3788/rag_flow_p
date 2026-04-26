import json
import re
from dataclasses import dataclass

import requests

from app.config import settings


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str
    confidence: float = 0.55


@dataclass
class ExtractedRelation:
    source_name: str
    target_name: str
    relation: str
    confidence: float = 0.45


ENTITY_PATTERN = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,3})\b")


def extract_graph_elements(chunk_text: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    if not chunk_text.strip():
        return [], []
    entities, relations = _extract_with_llm(chunk_text)
    if entities:
        return entities, relations
    fallback_entities = _extract_entities_rule_based(chunk_text)
    fallback_relations = _extract_relations_rule_based(chunk_text, fallback_entities)
    return fallback_entities, fallback_relations


def _extract_with_llm(chunk_text: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    prompt = _build_llm_extraction_prompt(chunk_text)
    model = settings.graphrag_extraction_model.strip() or settings.chat_model
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You extract graph entities and relations from text. "
                            "Return strict JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=settings.graphrag_extraction_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        content = str(payload.get("message", {}).get("content", "")).strip()
        return _parse_llm_payload(content)
    except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError):
        return [], []


def _build_llm_extraction_prompt(chunk_text: str) -> str:
    return (
        "Extract entities and directed relations from the text below.\n"
        "Return strict JSON with this exact shape:\n"
        '{\n'
        '  "entities":[{"name":"...","entity_type":"person|org|location|event|concept|other","description":"...","confidence":0.0}],\n'
        '  "relations":[{"source_name":"...","target_name":"...","relation":"...","confidence":0.0}]\n'
        '}\n'
        "Rules:\n"
        "- Keep entity names concise and canonical.\n"
        "- confidence must be in [0,1].\n"
        "- relation should be snake_case verb phrase (e.g. works_for, part_of, uses).\n"
        "- If none found, return empty arrays.\n\n"
        f"Text:\n{chunk_text}"
    )


def _parse_llm_payload(content: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    obj = _parse_json_object(content)
    raw_entities = obj.get("entities", [])
    raw_relations = obj.get("relations", [])
    if not isinstance(raw_entities, list) or not isinstance(raw_relations, list):
        return [], []

    entities: list[ExtractedEntity] = []
    seen_entities: set[str] = set()
    for item in raw_entities[:20]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if len(name) < 2:
            continue
        canonical = name.lower()
        if canonical in seen_entities:
            continue
        seen_entities.add(canonical)
        entity_type = _normalize_entity_type(str(item.get("entity_type", "concept")).strip())
        description = str(item.get("description", "")).strip() or f"Mentioned in text: {name}"
        confidence = _clamp_confidence(item.get("confidence"), default=0.65)
        entities.append(
            ExtractedEntity(
                name=name,
                entity_type=entity_type,
                description=description,
                confidence=confidence,
            )
        )
    if not entities:
        return [], []

    known_names = {entity.name.lower() for entity in entities}
    relations: list[ExtractedRelation] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for item in raw_relations[:40]:
        if not isinstance(item, dict):
            continue
        source_name = str(item.get("source_name", "")).strip()
        target_name = str(item.get("target_name", "")).strip()
        relation = _normalize_relation(str(item.get("relation", "")).strip())
        if not source_name or not target_name or not relation:
            continue
        if source_name.lower() not in known_names or target_name.lower() not in known_names:
            continue
        if source_name.lower() == target_name.lower():
            continue
        key = (source_name.lower(), target_name.lower(), relation)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append(
            ExtractedRelation(
                source_name=source_name,
                target_name=target_name,
                relation=relation,
                confidence=_clamp_confidence(item.get("confidence"), default=0.55),
            )
        )
    return entities, relations


def _parse_json_object(content: str) -> dict:
    if not content:
        return {}
    stripped = content.strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return {}
    parsed = json.loads(match.group(0))
    return parsed if isinstance(parsed, dict) else {}


def _normalize_entity_type(entity_type: str) -> str:
    lowered = entity_type.lower().strip()
    allowed = {"person", "org", "location", "event", "concept", "other"}
    return lowered if lowered in allowed else "concept"


def _normalize_relation(relation: str) -> str:
    cleaned = relation.lower().strip().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    return cleaned or "related_to"


def _clamp_confidence(value: object, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _extract_entities_rule_based(chunk_text: str) -> list[ExtractedEntity]:
    if not chunk_text.strip():
        return []
    seen: set[str] = set()
    extracted: list[ExtractedEntity] = []
    for match in ENTITY_PATTERN.finditer(chunk_text):
        raw = match.group(1).strip()
        if len(raw) < 3:
            continue
        canonical = raw.lower()
        if canonical in seen:
            continue
        seen.add(canonical)
        extracted.append(
            ExtractedEntity(
                name=raw,
                entity_type="concept",
                description=f"Mentioned in text: {raw}",
            )
        )
        if len(extracted) >= 12:
            break
    return extracted


def _extract_relations_rule_based(
    chunk_text: str, entities: list[ExtractedEntity]
) -> list[ExtractedRelation]:
    if len(entities) < 2:
        return []
    names = [entity.name for entity in entities[:8]]
    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str]] = set()
    for idx in range(len(names) - 1):
        source = names[idx]
        target = names[idx + 1]
        key = (source.lower(), target.lower())
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            ExtractedRelation(
                source_name=source,
                target_name=target,
                relation=_infer_relation(chunk_text, source, target),
            )
        )
    return relations


def _infer_relation(chunk_text: str, source: str, target: str) -> str:
    lowered = chunk_text.lower()
    source_lower = source.lower()
    target_lower = target.lower()
    window_index = lowered.find(source_lower)
    if window_index >= 0:
        window = lowered[window_index : window_index + 220]
        if "cause" in window or "because" in window:
            return "influences"
        if "part of" in window or "belongs to" in window:
            return "part_of"
        if "uses" in window or "utilize" in window:
            return "uses"
    if source_lower in lowered and target_lower in lowered:
        return "related_to"
    return "co_occurs_with"
