import requests
from fastapi import HTTPException

from app.config import settings


def _ollama_embeddings(input_text: str) -> list[float]:
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/embeddings",
            json={"model": settings.embedding_model, "prompt": input_text},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not embedding:
            raise HTTPException(status_code=500, detail="No embedding returned from Ollama")
        return embedding
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Ollama embedding error: {exc}") from exc

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [_ollama_embeddings(text) for text in texts]


def embed_query(query: str) -> list[float]:
    return _ollama_embeddings(query)
