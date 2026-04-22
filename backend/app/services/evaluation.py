import json
from typing import Any

import requests
from fastapi import HTTPException

from app.config import settings
from app.schemas import EvaluationScores


def evaluate_rag_output(
    query: str, answer: str, contexts: list[str], model: str | None = None
) -> EvaluationScores:
    prompt = _build_evaluation_prompt(query=query, answer=answer, contexts=contexts)
    raw = _call_judge_llm(prompt, model=model)
    parsed = _parse_scores(raw)
    return EvaluationScores(**parsed)


def _build_evaluation_prompt(query: str, answer: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(f"- {item}" for item in contexts) if contexts else "- (none)"
    return (
        "You are an evaluator for RAG outputs.\n"
        "Score each metric from 0.0 to 1.0:\n"
        "1) faithfulness: Is the answer grounded in provided contexts?\n"
        "2) relevance: Does the answer directly answer the query?\n"
        "3) context_precision: Are the provided contexts actually useful for the answer?\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{ "faithfulness": float, "relevance": float, "context_precision": float, "rationale": string }\n\n'
        f"Query:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Contexts:\n{context_text}"
    )


def _call_judge_llm(prompt: str, model: str | None = None) -> str:
    selected_model = model or settings.chat_model
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": selected_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a strict JSON evaluator. Return only JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "")
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation model error: {exc}") from exc


def _parse_scores(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if not candidate:
        raise HTTPException(status_code=500, detail="Evaluator returned empty output")

    # Handle models that wrap JSON in extra text.
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluator returned invalid JSON: {raw}",
        ) from exc

    required = {"faithfulness", "relevance", "context_precision"}
    if not required.issubset(payload.keys()):
        raise HTTPException(
            status_code=500,
            detail="Evaluator JSON missing required metric fields",
        )

    return {
        "faithfulness": _clamp01(payload.get("faithfulness", 0)),
        "relevance": _clamp01(payload.get("relevance", 0)),
        "context_precision": _clamp01(payload.get("context_precision", 0)),
        "rationale": str(payload.get("rationale", "")) if payload.get("rationale") else None,
    }


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=f"Invalid metric value: {value}") from exc
    if numeric < 0:
        return 0.0
    if numeric > 1:
        return 1.0
    return numeric
