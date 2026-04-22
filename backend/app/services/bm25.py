import json
from dataclasses import dataclass

import requests
from fastapi import HTTPException

from app.config import settings


@dataclass
class BM25Hit:
    chunk_id: int
    score: float


def build_dataset_index_name(dataset_id: int) -> str:
    return f"dataset_{dataset_id}_bm25"


def index_chunks(
    dataset_id: int,
    items: list[dict],
) -> None:
    if not items:
        return
    index_name = build_dataset_index_name(dataset_id)
    _ensure_index(index_name)

    ndjson_lines: list[str] = []
    for item in items:
        chunk_id = int(item["chunk_id"])
        ndjson_lines.append(json.dumps({"index": {"_index": index_name, "_id": str(chunk_id)}}))
        ndjson_lines.append(
            json.dumps(
                {
                    "chunk_id": chunk_id,
                    "content": str(item.get("content", "")),
                    "metadata": item.get("metadata") or {},
                }
            )
        )
    payload = "\n".join(ndjson_lines) + "\n"
    response = _es_request(
        "POST",
        "/_bulk",
        data=payload,
        headers={"Content-Type": "application/x-ndjson"},
    )
    body = response.json()
    if body.get("errors"):
        raise HTTPException(status_code=502, detail="Elasticsearch bulk indexing failed")


def bm25_search(query: str, dataset_id: int, top_k: int) -> list[BM25Hit]:
    index_name = build_dataset_index_name(dataset_id)
    response = _es_request(
        "POST",
        f"/{index_name}/_search",
        json={
            "size": max(1, top_k),
            "query": {
                "match": {
                    "content": query,
                }
            },
        },
        allow_404=True,
    )
    if response.status_code == 404:
        return []
    hits = response.json().get("hits", {}).get("hits", [])
    return [
        BM25Hit(chunk_id=int(hit.get("_source", {}).get("chunk_id")), score=float(hit.get("_score", 0.0)))
        for hit in hits
        if hit.get("_source", {}).get("chunk_id") is not None
    ]


def _ensure_index(index_name: str) -> None:
    exists = _es_request("HEAD", f"/{index_name}", allow_404=True)
    if exists.status_code == 200:
        return
    if exists.status_code != 404:
        exists.raise_for_status()
    _es_request(
        "PUT",
        f"/{index_name}",
        json={
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "integer"},
                    "content": {"type": "text"},
                    "metadata": {"type": "object", "enabled": True},
                }
            }
        },
    )


def _es_request(
    method: str,
    path: str,
    json: dict | None = None,
    data: str | None = None,
    headers: dict | None = None,
    allow_404: bool = False,
) -> requests.Response:
    auth: tuple[str, str] | None = None
    if settings.elasticsearch_username and settings.elasticsearch_password:
        auth = (settings.elasticsearch_username, settings.elasticsearch_password)
    try:
        response = requests.request(
            method=method,
            url=f"{settings.elasticsearch_url.rstrip('/')}{path}",
            json=json,
            data=data,
            headers=headers,
            auth=auth,
            timeout=settings.elasticsearch_timeout_seconds,
        )
        if allow_404 and response.status_code == 404:
            return response
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Elasticsearch error: {exc}") from exc
