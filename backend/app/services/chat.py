import requests
from fastapi import HTTPException

from app.config import settings
from app.schemas import SourceChunk


def build_prompt(query: str, sources: list[SourceChunk]) -> str:
    context_blocks = []
    for src in sources:
        context_blocks.append(
            f"[Doc: {src.document_name} | Chunk ID: {src.chunk_id}]\n{src.content}"
        )
    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."
    return (
        "You are a helpful RAG assistant. Use only the provided context when possible. "
        "If the answer is not in context, say that clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer with concise, accurate information."
    )


def generate_answer(query: str, sources: list[SourceChunk]) -> str:
    prompt = build_prompt(query, sources)
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.chat_model,
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
