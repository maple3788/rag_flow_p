import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from fastapi import HTTPException

from app.config import settings


@dataclass
class DenseHit:
    chunk_id: int
    score: float


@dataclass
class FileDenseHit:
    file_id: int
    score: float


def add_embeddings(dataset_id: int, embeddings: list[list[float]], chunk_ids: list[int]) -> None:
    if not embeddings:
        return
    if len(embeddings) != len(chunk_ids):
        raise HTTPException(status_code=500, detail="FAISS embeddings/chunk_ids mismatch")

    index_path, mapping_path = _dataset_paths(dataset_id)
    index, mapping = _load_or_create(index_path=index_path, mapping_path=mapping_path)

    vectors = _to_normalized_array(embeddings)
    index.add(vectors)
    mapping.extend(int(chunk_id) for chunk_id in chunk_ids)

    _save(index=index, mapping=mapping, index_path=index_path, mapping_path=mapping_path)


def dense_search(query_embedding: list[float], dataset_id: int, top_k: int) -> list[DenseHit]:
    index_path, mapping_path = _dataset_paths(dataset_id)
    if not index_path.exists() or not mapping_path.exists():
        return []

    index = faiss.read_index(str(index_path))
    mapping = _load_mapping(mapping_path)
    if index.ntotal == 0 or not mapping:
        return []

    query_vector = _to_normalized_array([query_embedding])
    limit = max(1, min(top_k, len(mapping)))
    scores, ids = index.search(query_vector, limit)
    hits: list[DenseHit] = []
    for score, vector_id in zip(scores[0], ids[0], strict=False):
        if vector_id < 0 or vector_id >= len(mapping):
            continue
        hits.append(DenseHit(chunk_id=int(mapping[vector_id]), score=float(score)))
    return hits


def add_file_summary_embedding(dataset_id: int, file_id: int, embedding: list[float]) -> None:
    index_path, mapping_path = _dataset_paths(dataset_id, namespace="summary")
    index, mapping = _load_or_create(index_path=index_path, mapping_path=mapping_path)
    index.add(_to_normalized_array([embedding]))
    mapping.append(int(file_id))
    _save(index=index, mapping=mapping, index_path=index_path, mapping_path=mapping_path)


def dense_search_file_summaries(query_embedding: list[float], dataset_id: int, top_k: int) -> list[FileDenseHit]:
    index_path, mapping_path = _dataset_paths(dataset_id, namespace="summary")
    if not index_path.exists() or not mapping_path.exists():
        return []

    index = faiss.read_index(str(index_path))
    mapping = _load_mapping(mapping_path)
    if index.ntotal == 0 or not mapping:
        return []

    query_vector = _to_normalized_array([query_embedding])
    limit = max(1, min(top_k, len(mapping)))
    scores, ids = index.search(query_vector, limit)
    hits: list[FileDenseHit] = []
    for score, vector_id in zip(scores[0], ids[0], strict=False):
        if vector_id < 0 or vector_id >= len(mapping):
            continue
        hits.append(FileDenseHit(file_id=int(mapping[vector_id]), score=float(score)))
    return hits


def _dataset_paths(dataset_id: int, namespace: str = "chunk") -> tuple[Path, Path]:
    base_dir = Path(settings.faiss_index_dir) / f"dataset_{dataset_id}"
    base_dir.mkdir(parents=True, exist_ok=True)
    if namespace == "summary":
        return (base_dir / "summary_index.faiss", base_dir / "summary_mapping.json")
    return (base_dir / "index.faiss", base_dir / "mapping.json")


def _load_or_create(index_path: Path, mapping_path: Path) -> tuple[faiss.Index, list[int]]:
    if index_path.exists() and mapping_path.exists():
        return faiss.read_index(str(index_path)), _load_mapping(mapping_path)
    index = faiss.IndexFlatIP(settings.embedding_dimension)
    mapping: list[int] = []
    return index, mapping


def _load_mapping(mapping_path: Path) -> list[int]:
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            return []
        return [int(value) for value in payload]
    except (json.JSONDecodeError, OSError, ValueError):
        return []


def _save(index: faiss.Index, mapping: list[int], index_path: Path, mapping_path: Path) -> None:
    faiss.write_index(index, str(index_path))
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")


def _to_normalized_array(vectors: list[list[float]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2 or arr.shape[1] != settings.embedding_dimension:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Invalid embedding shape for FAISS: expected (*, {settings.embedding_dimension}), "
                f"got {arr.shape}"
            ),
        )
    faiss.normalize_L2(arr)
    return arr
