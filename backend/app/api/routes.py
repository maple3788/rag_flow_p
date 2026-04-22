import json
from collections.abc import Generator
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Chunk, DataFile, Dataset, Evaluation
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DatasetChunkResponse,
    DatasetCreateRequest,
    DatasetFileResponse,
    DatasetResponse,
    DatasetUpdateRequest,
    EvaluationHistoryItem,
    EvaluationSummaryPoint,
    EvaluateRequest,
    EvaluateResponse,
    UploadResponse,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from app.services.chat import generate_answer, stream_answer_tokens
from app.services.bm25 import index_chunks
from app.services.document_parser import parse_uploaded_file
from app.services.embeddings import embed_texts
from app.services.evaluation import evaluate_rag_output
from app.services.faiss_index import add_embeddings
from app.services.dataset_config import resolve_dataset_config
from app.services.query_ops import rerank_sources, rewrite_query
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
    default_dataset = db.query(Dataset).filter(Dataset.name == "default").first()
    if default_dataset is None:
        default_dataset = Dataset(
            name="default",
            description="Auto-created default dataset",
            config={},
        )
        db.add(default_dataset)
        db.flush()
    dataset_config = resolve_dataset_config(default_dataset.config)
    chunks = split_text_recursive(
        text=text,
        chunk_size=int(dataset_config["chunk_size"]),
        chunk_overlap=int(dataset_config["chunk_overlap"]),
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks were generated")

    vectors = embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    data_file = DataFile(
        dataset_id=default_dataset.id,
        filename=file.filename or "uploaded_document",
        raw_text=text,
        file_metadata={},
    )
    db.add(data_file)
    db.flush()

    chunk_rows = []
    for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors, strict=False)):
        chunk_rows.append(
            Chunk(
                file_id=data_file.id,
                dataset_id=default_dataset.id,
                content=chunk_text,
                embedding=vector,
                chunk_metadata={"chunk_index": idx},
            )
        )
    db.add_all(chunk_rows)
    db.flush()
    add_embeddings(
        dataset_id=default_dataset.id,
        embeddings=vectors,
        chunk_ids=[chunk.id for chunk in chunk_rows],
    )
    index_chunks(
        dataset_id=default_dataset.id,
        items=[
            {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.chunk_metadata,
            }
            for chunk in chunk_rows
        ],
    )
    db.commit()

    return UploadResponse(
        document_id=data_file.id,
        file_name=data_file.filename,
        chunks_indexed=len(chunk_rows),
    )


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    _validate_model_option(request.model)
    conversation_id = request.conversation_id or _new_conversation_id()

    retrieval_query = request.query
    dataset_config = _get_dataset_config(db=db, dataset_id=request.dataset_id)
    use_query_rewrite = request.enable_query_rewrite or bool(dataset_config["enable_query_rewrite"])
    if use_query_rewrite:
        retrieval_query = rewrite_query(request.query, model=request.model)

    final_k = request.k or int(dataset_config["final_k"]) or settings.top_k
    sources = retrieve_similar_chunks(
        db=db,
        query=retrieval_query,
        k=final_k,
        dataset_id=request.dataset_id,
        top_k_bm25=int(dataset_config["top_k_bm25"]),
        top_k_dense=int(dataset_config["top_k_dense"]),
        fusion_method=str(dataset_config["fusion_method"]),
        rerank_enabled=bool(dataset_config["rerank_enabled"]),
        rerank_model=str(dataset_config["rerank_model"]),
    )
    if request.enable_rerank and request.dataset_id is None:
        sources = rerank_sources(request.query, sources)
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
        rewritten_query=retrieval_query if retrieval_query != request.query else None,
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

    retrieval_query = request.query
    dataset_config = _get_dataset_config(db=db, dataset_id=request.dataset_id)
    use_query_rewrite = request.enable_query_rewrite or bool(dataset_config["enable_query_rewrite"])
    if use_query_rewrite:
        retrieval_query = rewrite_query(request.query, model=request.model)

    final_k = request.k or int(dataset_config["final_k"]) or settings.top_k
    sources = retrieve_similar_chunks(
        db=db,
        query=retrieval_query,
        k=final_k,
        dataset_id=request.dataset_id,
        top_k_bm25=int(dataset_config["top_k_bm25"]),
        top_k_dense=int(dataset_config["top_k_dense"]),
        fusion_method=str(dataset_config["fusion_method"]),
        rerank_enabled=bool(dataset_config["rerank_enabled"]),
        rerank_model=str(dataset_config["rerank_model"]),
    )
    if request.enable_rerank and request.dataset_id is None:
        sources = rerank_sources(request.query, sources)

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
                    "rewritten_query": (
                        retrieval_query if retrieval_query != request.query else None
                    ),
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


@router.post("/datasets", response_model=DatasetResponse)
def create_dataset(
    request: DatasetCreateRequest,
    db: Session = Depends(get_db),
) -> DatasetResponse:
    existing = db.query(Dataset).filter(Dataset.name == request.name).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Dataset name already exists")
    dataset = Dataset(
        name=request.name.strip(),
        description=request.description,
        config=request.config,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        config=dataset.config,
        created_at=dataset.created_at,
    )


@router.get("/datasets", response_model=list[DatasetResponse])
def list_datasets(db: Session = Depends(get_db)) -> list[DatasetResponse]:
    datasets = db.query(Dataset).order_by(desc(Dataset.created_at)).all()
    return [
        DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            config=dataset.config,
            created_at=dataset.created_at,
        )
        for dataset in datasets
    ]


@router.get("/datasets/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)) -> DatasetResponse:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        config=dataset.config,
        created_at=dataset.created_at,
    )


@router.patch("/datasets/{dataset_id}", response_model=DatasetResponse)
def update_dataset(
    dataset_id: int,
    request: DatasetUpdateRequest,
    db: Session = Depends(get_db),
) -> DatasetResponse:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if request.description is not None:
        dataset.description = request.description
    if request.config is not None:
        dataset.config = request.config
    db.commit()
    db.refresh(dataset)
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        config=dataset.config,
        created_at=dataset.created_at,
    )


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)) -> dict:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    db.delete(dataset)
    db.commit()
    return {"status": "deleted", "dataset_id": dataset_id}


@router.post("/datasets/{dataset_id}/files", response_model=DatasetFileResponse)
async def upload_dataset_file(
    dataset_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DatasetFileResponse:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    text = await parse_uploaded_file(file)
    dataset_config = resolve_dataset_config(dataset.config)
    chunks = split_text_recursive(
        text=text,
        chunk_size=int(dataset_config["chunk_size"]),
        chunk_overlap=int(dataset_config["chunk_overlap"]),
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks were generated")

    vectors = embed_texts(chunks)
    if len(vectors) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    data_file = DataFile(
        dataset_id=dataset.id,
        filename=file.filename or "uploaded_document",
        raw_text=text,
        file_metadata={},
    )
    db.add(data_file)
    db.flush()

    chunk_rows = []
    for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors, strict=False)):
        chunk_rows.append(
            Chunk(
                file_id=data_file.id,
                dataset_id=dataset.id,
                content=chunk_text,
                embedding=vector,
                chunk_metadata={"chunk_index": idx},
            )
        )
    db.add_all(chunk_rows)
    db.flush()
    add_embeddings(
        dataset_id=dataset.id,
        embeddings=vectors,
        chunk_ids=[chunk.id for chunk in chunk_rows],
    )
    index_chunks(
        dataset_id=dataset.id,
        items=[
            {
                "chunk_id": chunk.id,
                "content": chunk.content,
                "metadata": chunk.chunk_metadata,
            }
            for chunk in chunk_rows
        ],
    )
    db.commit()
    db.refresh(data_file)
    return DatasetFileResponse(
        id=data_file.id,
        dataset_id=data_file.dataset_id,
        filename=data_file.filename,
        raw_text=data_file.raw_text,
        metadata=data_file.file_metadata,
    )


@router.get("/datasets/{dataset_id}/files", response_model=list[DatasetFileResponse])
def list_dataset_files(
    dataset_id: int, db: Session = Depends(get_db)
) -> list[DatasetFileResponse]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    rows = db.query(DataFile).filter(DataFile.dataset_id == dataset_id).order_by(DataFile.id.desc()).all()
    return [
        DatasetFileResponse(
            id=row.id,
            dataset_id=row.dataset_id,
            filename=row.filename,
            raw_text=row.raw_text,
            metadata=row.file_metadata,
        )
        for row in rows
    ]


@router.delete("/datasets/{dataset_id}/files/{file_id}")
def delete_dataset_file(
    dataset_id: int,
    file_id: int,
    db: Session = Depends(get_db),
) -> dict:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    data_file = (
        db.query(DataFile)
        .filter(DataFile.id == file_id, DataFile.dataset_id == dataset_id)
        .first()
    )
    if data_file is None:
        raise HTTPException(status_code=404, detail="File not found in dataset")
    db.delete(data_file)
    db.commit()
    return {"status": "deleted", "dataset_id": dataset_id, "file_id": file_id}


@router.get("/datasets/{dataset_id}/chunks", response_model=list[DatasetChunkResponse])
def list_dataset_chunks(
    dataset_id: int,
    file_id: int | None = None,
    db: Session = Depends(get_db),
) -> list[DatasetChunkResponse]:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    query = db.query(Chunk).filter(Chunk.dataset_id == dataset_id)
    if file_id is not None:
        query = query.filter(Chunk.file_id == file_id)
    rows = query.order_by(Chunk.id.desc()).all()
    return [
        DatasetChunkResponse(
            id=row.id,
            file_id=row.file_id,
            dataset_id=row.dataset_id,
            content=row.content,
            metadata=row.chunk_metadata,
        )
        for row in rows
    ]


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


def _get_dataset_config(db: Session, dataset_id: int | None) -> dict:
    if dataset_id is None:
        return resolve_dataset_config({})
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return resolve_dataset_config(dataset.config)
