import json
from collections.abc import Generator

import requests
from fastapi import HTTPException

from app.config import settings
from app.schemas import SourceChunk


def build_prompt(query: str, sources: list[SourceChunk]) -> str:
    context_blocks = []
    for idx, src in enumerate(sources, start=1):
        context_blocks.append(
            f"[{idx}] [Dataset: {src.dataset_id} | File: {src.filename} | Chunk ID: {src.chunk_id}]\n{src.content}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
    return (
        "You are a helpful RAG assistant. Use only the provided context when possible. "
        "If the answer is not in context, say that clearly.\n"
        "When you use a fact from context, add inline citation markers like [1], [2] at the end of the sentence.\n"
        "Only cite source numbers that exist in the provided context blocks.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer with concise, accurate information and include inline citations."
    )


def generate_answer(
    query: str, sources: list[SourceChunk], model: str | None = None
) -> str:
    prompt = build_prompt(query, sources)
    selected_model = model or settings.chat_model
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": selected_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer questions using retrieved document context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "")
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Ollama chat error: {exc}") from exc


def stream_answer_tokens(
    query: str, sources: list[SourceChunk], model: str | None = None
) -> Generator[str, None, None]:
    prompt = build_prompt(query, sources)
    selected_model = model or settings.chat_model
    try:
        with requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": selected_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You answer questions using retrieved document context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {"temperature": 0.2},
            },
            timeout=180,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = payload.get("message", {}).get("content", "")
                if chunk:
                    yield chunk
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Ollama stream error: {exc}") from exc
