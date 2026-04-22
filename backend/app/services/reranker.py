import json
import math

import requests

from app.config import settings
from app.schemas import SourceChunk


def rerank_fused_chunks(
    query: str,
    chunks: list[SourceChunk],
    top_k: int,
    model_name: str | None = None,
) -> list[SourceChunk]:
    if not chunks:
        return chunks

    strategy = settings.reranker_strategy.strip().lower()
    if strategy not in {"auto", "cross_encoder", "llm"}:
        strategy = "auto"

    if strategy in {"auto", "cross_encoder"}:
        reranked = _rerank_with_cross_encoder(
            query=query,
            chunks=chunks,
            top_k=top_k,
            model_name=model_name,
        )
        if reranked is not None:
            return reranked
        if strategy == "cross_encoder":
            return chunks[:top_k]

    reranked = _rerank_with_llm(
        query=query,
        chunks=chunks,
        top_k=top_k,
        model_name=model_name,
    )
    if reranked is not None:
        return reranked
    return chunks[:top_k]


def _rerank_with_cross_encoder(
    query: str,
    chunks: list[SourceChunk],
    top_k: int,
    model_name: str | None = None,
) -> list[SourceChunk] | None:
    model = _get_cross_encoder(model_name=model_name)
    if model is None:
        return None
    pairs = [[query, chunk.content] for chunk in chunks]
    logits = model.predict(pairs)
    # Cross-encoder outputs are often logits (can be negative). Convert to [0, 1].
    scores = [1.0 / (1.0 + math.exp(-float(logit))) for logit in logits]
    normalized_scores = _min_max_normalize(scores)
    scored = list(zip(normalized_scores, chunks, strict=False))
    scored.sort(key=lambda item: float(item[0]), reverse=True)
    return [_with_score(chunk=chunk, score=float(score)) for score, chunk in scored[:top_k]]


def _rerank_with_llm(
    query: str,
    chunks: list[SourceChunk],
    top_k: int,
    model_name: str | None = None,
) -> list[SourceChunk] | None:
    chunk_payload = [
        {"chunk_id": chunk.chunk_id, "content": chunk.content[:1800]}
        for chunk in chunks
    ]
    prompt = (
        "Given a user query and retrieved chunks, score each chunk's relevance from 0.0 to 1.0.\n"
        "Return JSON only with this schema: "
        '{"scores":[{"chunk_id":123,"score":0.85}]}\n'
        f"Query: {query}\n"
        f"Chunks: {json.dumps(chunk_payload, ensure_ascii=True)}"
    )
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": model_name or settings.reranker_model,
                "messages": [
                    {"role": "system", "content": "You are a precise retrieval reranker. Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=180,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        parsed = _safe_parse_json(content)
        score_items = parsed.get("scores", [])
        if not isinstance(score_items, list):
            return None
        score_by_id: dict[int, float] = {}
        for item in score_items:
            try:
                chunk_id = int(item.get("chunk_id"))
                score = float(item.get("score", 0.0))
                score_by_id[chunk_id] = score
            except (TypeError, ValueError, AttributeError):
                continue
        scored_chunks = [
            (score_by_id.get(chunk.chunk_id, float(chunk.score)), chunk)
            for chunk in chunks
        ]
        normalized = _min_max_normalize([score for score, _ in scored_chunks])
        scored_chunks = [
            (normalized_score, chunk)
            for normalized_score, (_, chunk) in zip(normalized, scored_chunks, strict=False)
        ]
        scored_chunks.sort(key=lambda item: item[0], reverse=True)
        return [
            _with_score(chunk=chunk, score=score)
            for score, chunk in scored_chunks[:top_k]
        ]
    except requests.RequestException:
        return None


def _safe_parse_json(raw: str) -> dict:
    candidate = raw.strip()
    if not candidate:
        return {}
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _with_score(chunk: SourceChunk, score: float) -> SourceChunk:
    return SourceChunk(
        dataset_id=chunk.dataset_id,
        file_id=chunk.file_id,
        filename=chunk.filename,
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        metadata=chunk.metadata,
        score=score,
    )


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    span = high - low
    if span <= 1e-9:
        return [0.5 for _ in scores]
    return [(score - low) / span for score in scores]


_cross_encoders_by_model: dict[str, object] = {}
_cross_encoder_failed_models: set[str] = set()


def _get_cross_encoder(model_name: str | None = None):
    selected_model = (model_name or settings.cross_encoder_model).strip()
    if not selected_model:
        selected_model = settings.cross_encoder_model

    if selected_model in _cross_encoders_by_model:
        return _cross_encoders_by_model[selected_model]
    if selected_model in _cross_encoder_failed_models:
        return None

    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        loaded = CrossEncoder(selected_model)
        _cross_encoders_by_model[selected_model] = loaded
        return loaded
    except Exception:
        _cross_encoder_failed_models.add(selected_model)
        return None
