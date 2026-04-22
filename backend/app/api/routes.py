import json
from collections.abc import Generator
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Chunk, Document, Evaluation
from app.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluationHistoryItem,
    EvaluationSummaryPoint,
    EvaluateRequest,
    EvaluateResponse,
    UploadResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from app.services.chat import generate_answer, stream_answer_tokens
from app.services.document_parser import parse_uploaded_file
from app.services.embeddings import embed_texts
from app.services.evaluation import evaluate_rag_output
from app.services.retrieval import retrieve_similar_chunks
from app.services.text_splitter import split_text_recursive
from app.services.workflow.engine import run_workflow

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    text = await parse_uploaded_file(file)
    chunks = split_text_recursive(text=text, chunk_size=500, chunk_overlap=50)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks were generated")

    vectors = embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    document = Document(name=file.filename or "uploaded_document")
    db.add(document)
    db.flush()

    chunk_rows = []
    for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors, strict=False)):
        chunk_rows.append(
            Chunk(
                document_id=document.id,
                content=chunk_text,
                embedding=vector,
                chunk_metadata={"chunk_index": idx},
            )
        )
    db.add_all(chunk_rows)
    db.commit()

    return UploadResponse(
        document_id=document.id,
        file_name=document.name,
        chunks_indexed=len(chunk_rows),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    _validate_model_option(request.model)
    conversation_id = request.conversation_id or _new_conversation_id()

    k = request.k or settings.top_k
    sources = retrieve_similar_chunks(db=db, query=request.query, k=k)
    answer = generate_answer(query=request.query, sources=sources, model=request.model)
    evaluation = evaluate_rag_output(
        query=request.query,
        answer=answer,
        contexts=[source.content for source in sources],
        model=request.model,
    )

    db.add(
        Evaluation(
            conversation_id=conversation_id,
            query=request.query,
            answer=answer,
            scores=evaluation.model_dump(),
        )
    )
    db.commit()

    return ChatResponse(
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
        evaluation=evaluation,
    )


@router.post("/chat/stream")
def chat_stream(request: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    _validate_model_option(request.model)
    conversation_id = request.conversation_id or _new_conversation_id()

    k = request.k or settings.top_k
    sources = retrieve_similar_chunks(db=db, query=request.query, k=k)

    def event_stream() -> Generator[str, None, None]:
        answer_parts: list[str] = []
        for token in stream_answer_tokens(
            query=request.query,
            sources=sources,
            model=request.model,
        ):
            answer_parts.append(token)
            yield json.dumps({"type": "token", "content": token}) + "\n"

        answer = "".join(answer_parts).strip()
        evaluation = evaluate_rag_output(
            query=request.query,
            answer=answer,
            contexts=[source.content for source in sources],
            model=request.model,
        )

        db.add(
            Evaluation(
                conversation_id=conversation_id,
                query=request.query,
                answer=answer,
                scores=evaluation.model_dump(),
            )
        )
        db.commit()

        yield (
            json.dumps(
                {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "answer": answer,
                    "sources": [source.model_dump() for source in sources],
                    "evaluation": evaluation.model_dump(),
                }
            )
            + "\n"
        )

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest, db: Session = Depends(get_db)) -> EvaluateResponse:
    _validate_model_option(request.model)
    scores = evaluate_rag_output(
        query=request.query,
        answer=request.answer,
        contexts=request.contexts,
        model=request.model,
    )
    db.add(
        Evaluation(
            conversation_id=request.conversation_id or _new_conversation_id(),
            query=request.query,
            answer=request.answer,
            scores=scores.model_dump(),
        )
    )
    db.commit()
    return EvaluateResponse(scores=scores)


@router.get("/evaluations", response_model=list[EvaluationHistoryItem])
def get_evaluations(
    db: Session = Depends(get_db),
    limit: int = 20,
    conversation_id: str | None = None,
) -> list[EvaluationHistoryItem]:
    bounded_limit = max(1, min(limit, 100))
    query = db.query(Evaluation)
    if conversation_id:
        query = query.filter(Evaluation.conversation_id == conversation_id)
    rows = query.order_by(desc(Evaluation.created_at)).limit(bounded_limit).all()
    return [
        EvaluationHistoryItem(
            id=row.id,
            conversation_id=row.conversation_id,
            query=row.query,
            answer=row.answer,
            scores=row.scores,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


@router.get("/evaluations/summary", response_model=list[EvaluationSummaryPoint])
def get_evaluation_summary(
    db: Session = Depends(get_db),
    limit: int = 200,
    conversation_id: str | None = None,
) -> list[EvaluationSummaryPoint]:
    bounded_limit = max(1, min(limit, 500))
    query = db.query(Evaluation)
    if conversation_id:
        query = query.filter(Evaluation.conversation_id == conversation_id)
    rows = query.order_by(desc(Evaluation.created_at)).limit(bounded_limit).all()
    if not rows:
        return []

    buckets: dict[str, list[Evaluation]] = {}
    for row in sorted(rows, key=lambda item: item.created_at):
        period = row.created_at.strftime("%Y-%m-%d")
        buckets.setdefault(period, []).append(row)

    summary: list[EvaluationSummaryPoint] = []
    for period, entries in buckets.items():
        faithfulness = _avg_metric(entries, "faithfulness")
        relevance = _avg_metric(entries, "relevance")
        context_precision = _avg_metric(entries, "context_precision")
        summary.append(
            EvaluationSummaryPoint(
                period=period,
                faithfulness=faithfulness,
                relevance=relevance,
                context_precision=context_precision,
                count=len(entries),
            )
        )
    return summary


@router.post("/workflow/run", response_model=WorkflowRunResponse)
def run_workflow_api(
    request: WorkflowRunRequest, db: Session = Depends(get_db)
) -> WorkflowRunResponse:
    return run_workflow(payload=request, db=db)


def _validate_model_option(model: str | None) -> None:
    if model is None:
        return
    if model not in settings.chat_model_options:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported model '{model}'. "
                f"Allowed models: {', '.join(settings.chat_model_options)}"
            ),
        )


def _new_conversation_id() -> str:
    return uuid4().hex


def _avg_metric(rows: list[Evaluation], metric: str) -> float:
    values: list[float] = []
    for row in rows:
        value = row.scores.get(metric, 0)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(0.0)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)
