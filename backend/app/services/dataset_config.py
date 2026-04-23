from typing import Any


def default_dataset_config() -> dict[str, Any]:
    return {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "use_summary": False,
        "summarization_mode": "single",
        "file_router_top_k": 8,
        "enable_query_rewrite": False,
        "rerank_enabled": True,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }


def resolve_dataset_config(raw_config: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_config or {}
    defaults = default_dataset_config()

    chunk_size = _as_int(raw.get("chunk_size"), defaults["chunk_size"], minimum=100, maximum=4000)
    chunk_overlap = _as_int(raw.get("chunk_overlap"), defaults["chunk_overlap"], minimum=0, maximum=1000)
    use_summary = _as_bool(raw.get("use_summary"), defaults["use_summary"])
    summarization_mode = str(raw.get("summarization_mode") or defaults["summarization_mode"]).strip().lower()
    if summarization_mode not in {"single", "hierarchical", "iterative"}:
        summarization_mode = "single"
    file_router_top_k = _as_int(raw.get("file_router_top_k"), defaults["file_router_top_k"], minimum=1, maximum=30)
    enable_query_rewrite = _as_bool(raw.get("enable_query_rewrite"), defaults["enable_query_rewrite"])
    rerank_enabled = _as_bool(raw.get("rerank_enabled"), defaults["rerank_enabled"])
    rerank_model = str(raw.get("rerank_model") or defaults["rerank_model"]).strip()
    if not rerank_model:
        rerank_model = defaults["rerank_model"]

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_summary": use_summary,
        "summarization_mode": summarization_mode,
        "file_router_top_k": file_router_top_k,
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
