from typing import Any


def default_dataset_config() -> dict[str, Any]:
    return {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "final_k": 5,
        "top_k_bm25": 10,
        "top_k_dense": 10,
        "fusion_method": "rrf",
        "enable_query_rewrite": False,
        "rerank_enabled": True,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }


def resolve_dataset_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_config or {}
    defaults = default_dataset_config()

    chunk_size = _as_int(raw.get("chunk_size"), defaults["chunk_size"], minimum=100, maximum=4000)
    chunk_overlap = _as_int(raw.get("chunk_overlap"), defaults["chunk_overlap"], minimum=0, maximum=1000)
    final_k = _as_int(raw.get("final_k"), defaults["final_k"], minimum=1, maximum=50)
    top_k_bm25 = _as_int(raw.get("top_k_bm25"), defaults["top_k_bm25"], minimum=1, maximum=200)
    top_k_dense = _as_int(raw.get("top_k_dense"), defaults["top_k_dense"], minimum=1, maximum=200)
    fusion_method = str(raw.get("fusion_method") or defaults["fusion_method"]).strip().lower()
    if fusion_method != "rrf":
        fusion_method = "rrf"
    enable_query_rewrite = _as_bool(raw.get("enable_query_rewrite"), defaults["enable_query_rewrite"])
    rerank_enabled = _as_bool(raw.get("rerank_enabled"), defaults["rerank_enabled"])
    rerank_model = str(raw.get("rerank_model") or defaults["rerank_model"]).strip()
    if not rerank_model:
        rerank_model = defaults["rerank_model"]

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "final_k": final_k,
        "top_k_bm25": top_k_bm25,
        "top_k_dense": top_k_dense,
        "fusion_method": fusion_method,
        "enable_query_rewrite": enable_query_rewrite,
        "rerank_enabled": rerank_enabled,
        "rerank_model": rerank_model,
    }


def _as_int(value: Any, fallback: int, minimum: int, maximum: int) -> int:
    parsed = fallback
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            parsed = fallback
    return max(minimum, min(maximum, parsed))


def _as_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return fallback
