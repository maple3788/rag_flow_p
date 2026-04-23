import requests
from fastapi import HTTPException

from app.config import settings

ALLOWED_SUMMARY_MODES = {"single", "hierarchical", "iterative"}


def summarize_document(text: str, mode: str = "single", model: str | None = None) -> str:
    normalized_mode = (mode or "single").strip().lower()
    if normalized_mode not in ALLOWED_SUMMARY_MODES:
        normalized_mode = "single"
    source = (text or "").strip()
    if not source:
        return ""
    if len(source) > 20000:
        source = source[:20000]

    if normalized_mode == "single":
        return _summarize_single(source=source, model=model)
    if normalized_mode == "hierarchical":
        return _summarize_hierarchical(source=source, model=model)
    return _summarize_iterative(source=source, model=model)


def _summarize_single(source: str, model: str | None = None) -> str:
    prompt = (
        "Summarize this document for retrieval routing.\n"
        "Return concise plain text with key entities, topics, and terminology.\n"
        "Do not include markdown headings."
    )
    return _ollama_summarize(prompt=prompt, source=source, model=model)


def _summarize_hierarchical(source: str, model: str | None = None) -> str:
    chunks = _chunk_text(source, chunk_size=3500)
    partials = [
        _ollama_summarize(
            prompt=(
                "Summarize this section for retrieval routing.\n"
                "Focus on entities, terminology, and key claims."
            ),
            source=chunk,
            model=model,
        )
        for chunk in chunks
    ]
    merged = "\n\n".join(partials)
    return _ollama_summarize(
        prompt=(
            "Merge these section summaries into one retrieval-oriented summary.\n"
            "Keep it compact and high-signal."
        ),
        source=merged,
        model=model,
    )


def _summarize_iterative(source: str, model: str | None = None) -> str:
    chunks = _chunk_text(source, chunk_size=3000)
    memory = ""
    for chunk in chunks:
        seed = memory or "(none)"
        memory = _ollama_summarize(
            prompt=(
                "Update the running retrieval summary using the new section.\n"
                "Keep the summary concise and include unique terms/entities.\n"
                f"Current summary:\n{seed}"
            ),
            source=chunk,
            model=model,
        )
    return memory


def _ollama_summarize(prompt: str, source: str, model: str | None = None) -> str:
    selected_model = model or settings.chat_model
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": "You produce concise retrieval summaries."},
                    {"role": "user", "content": f"{prompt}\n\nDocument:\n{source}"},
                ],
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        summary = str(payload.get("message", {}).get("content", "")).strip()
        return summary
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Document summarization error: {exc}") from exc


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    out: list[str] = []
    for idx in range(0, len(text), chunk_size):
        out.append(text[idx : idx + chunk_size])
    return out
