import json
import re

import requests
from fastapi import HTTPException

from app.config import settings
from app.schemas import SourceChunk


def rewrite_query(query: str, model: str | None = None) -> str:
    selected_model = model or settings.chat_model
    prompt = (
        "Rewrite the user query to improve document retrieval quality.\n"
        "Keep intent unchanged, make it concise and keyword-rich.\n"
        "Return only the rewritten query text."
    )
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": "You optimize search queries."},
                    {"role": "user", "content": f"{prompt}\n\nQuery: {query}"},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        rewritten = str(payload.get("message", {}).get("content", "")).strip()
        return rewritten or query
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Query rewrite error: {exc}") from exc


def rerank_sources(query: str, sources: list[SourceChunk]) -> list[SourceChunk]:
    if not sources:
        return sources
    query_terms = _normalize_terms(query)
    scored: list[tuple[float, SourceChunk]] = []
    for src in sources:
        content_terms = _normalize_terms(src.content)
        overlap = (
            len(query_terms.intersection(content_terms)) / max(1, len(query_terms))
            if query_terms
            else 0.0
        )
        # Lower vector distance is better. Convert to a quality score.
        vector_quality = 1.0 / (1.0 + max(src.score, 0.0))
        final_score = 0.65 * overlap + 0.35 * vector_quality
        scored.append((final_score, src))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [src for _, src in scored]


def _normalize_terms(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    terms = {token for token in cleaned.split() if token and len(token) > 2}
    return terms
