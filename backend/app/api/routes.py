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
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    RetrievalStageHit,
    RetrievalFileStageHit,
    ChatRetrievalDebug,
)
from app.services.chat import generate_answer, stream_answer_tokens
from app.services.bm25 import index_chunks, index_file_summary
from app.services.document_parser import parse_uploaded_file
from app.services.embeddings import embed_texts
from app.services.evaluation import evaluate_rag_output
from app.services.faiss_index import add_embeddings, add_file_summary_embedding
from app.services.dataset_config import resolve_dataset_config
from app.services.query_ops import rerank_sources, rewrite_query
from app.services.retrieval import build_dataset_retrieval_debug, retrieve_similar_chunks
from app.services.summarization import summarize_document
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
    summary_text = ""
    if bool(dataset_config["use_summary"]):
        summary_text = summarize_document(
            text=text,
            mode=str(dataset_config["summarization_mode"]),
            model=settings.chat_model,
        )
        data_file.file_metadata = {
            **(data_file.file_metadata or {}),
            "summary": summary_text,
            "summarization_mode": str(dataset_config["summarization_mode"]),
        }

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
                "file_id": chunk.file_id,
                "content": chunk.content,
                "metadata": chunk.chunk_metadata,
            }
            for chunk in chunk_rows
        ],
    )
    if bool(dataset_config["use_summary"]) and summary_text.strip():
        index_file_summary(
            dataset_id=default_dataset.id,
            file_id=data_file.id,
            filename=data_file.filename,
            summary=summary_text,
            metadata=data_file.file_metadata,
        )
        summary_embedding = embed_texts([summary_text])[0]
        add_file_summary_embedding(
            dataset_id=default_dataset.id,
            file_id=data_file.id,
            embedding=summary_embedding,
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

    final_top_k = request.final_top_k or request.k or settings.top_k
    top_k_bm25 = request.top_k_bm25 or settings.top_k
    top_k_dense = request.top_k_dense or settings.top_k
    top_k_bm25 = max(final_top_k, top_k_bm25)
    top_k_dense = max(final_top_k, top_k_dense)
    retrieval_debug: ChatRetrievalDebug | None = None
    if request.dataset_id is not None:
        dataset_has_summary = _dataset_has_summary(db=db, dataset_id=request.dataset_id)
        requested_use_summary = (
            bool(request.use_summary)
            if request.use_summary is not None
            else bool(dataset_config["use_summary"])
        )
        effective_use_summary = requested_use_summary and dataset_has_summary
        summary_top_k = max(1, request.summary_top_k or int(dataset_config["file_router_top_k"]))
        summary_candidate_k = max(
            summary_top_k,
            request.summary_candidate_k or max(top_k_bm25, top_k_dense),
        )
        pipeline = build_dataset_retrieval_debug(
            db=db,
            query=retrieval_query,
            dataset_id=request.dataset_id,
            final_k=final_top_k,
            top_k_bm25=top_k_bm25,
            top_k_dense=top_k_dense,
            fusion_method="rrf",
            rerank_enabled=bool(dataset_config["rerank_enabled"]),
            rerank_model=str(dataset_config["rerank_model"]),
            use_summary=effective_use_summary,
            file_router_top_k=summary_top_k,
            summary_candidate_k=summary_candidate_k,
        )
        sources = pipeline["final_sources"]
        if not sources and retrieval_query != request.query:
            retrieval_query = request.query
            pipeline = build_dataset_retrieval_debug(
                db=db,
                query=retrieval_query,
                dataset_id=request.dataset_id,
                final_k=final_top_k,
                top_k_bm25=top_k_bm25,
                top_k_dense=top_k_dense,
                fusion_method="rrf",
                rerank_enabled=bool(dataset_config["rerank_enabled"]),
                rerank_model=str(dataset_config["rerank_model"]),
                use_summary=effective_use_summary,
                file_router_top_k=summary_top_k,
                summary_candidate_k=summary_candidate_k,
            )
            sources = pipeline["final_sources"]
        retrieval_debug = _to_chat_retrieval_debug(
            dataset_id=request.dataset_id,
            original_query=request.query,
            rewritten_query=retrieval_query if retrieval_query != request.query else None,
            used_query=retrieval_query,
            config={
                "top_k_bm25": top_k_bm25,
                "top_k_dense": top_k_dense,
                "final_top_k": final_top_k,
                "fusion_method": "rrf",
                "use_summary": effective_use_summary,
                "use_summary_requested": requested_use_summary,
                "dataset_has_summary": dataset_has_summary,
                "summarization_mode": str(dataset_config["summarization_mode"]),
                "file_router_top_k": summary_top_k,
                "summary_candidate_k": summary_candidate_k,
                "rerank_enabled": bool(dataset_config["rerank_enabled"]),
                "rerank_model": str(dataset_config["rerank_model"]),
            },
            pipeline=pipeline,
        )
    else:
        sources = retrieve_similar_chunks(
            db=db,
            query=request.query,
            k=final_top_k,
            dataset_id=None,
            top_k_bm25=top_k_bm25,
            top_k_dense=top_k_dense,
            fusion_method="rrf",
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
        retrieval_debug=retrieval_debug,
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

    final_top_k = request.final_top_k or request.k or settings.top_k
    top_k_bm25 = request.top_k_bm25 or settings.top_k
    top_k_dense = request.top_k_dense or settings.top_k
    top_k_bm25 = max(final_top_k, top_k_bm25)
    top_k_dense = max(final_top_k, top_k_dense)
    retrieval_debug: ChatRetrievalDebug | None = None
    if request.dataset_id is not None:
        dataset_has_summary = _dataset_has_summary(db=db, dataset_id=request.dataset_id)
        requested_use_summary = (
            bool(request.use_summary)
            if request.use_summary is not None
            else bool(dataset_config["use_summary"])
        )
        effective_use_summary = requested_use_summary and dataset_has_summary
        summary_top_k = max(1, request.summary_top_k or int(dataset_config["file_router_top_k"]))
        summary_candidate_k = max(
            summary_top_k,
            request.summary_candidate_k or max(top_k_bm25, top_k_dense),
        )
        pipeline = build_dataset_retrieval_debug(
            db=db,
            query=retrieval_query,
            dataset_id=request.dataset_id,
            final_k=final_top_k,
            top_k_bm25=top_k_bm25,
            top_k_dense=top_k_dense,
            fusion_method="rrf",
            rerank_enabled=bool(dataset_config["rerank_enabled"]),
            rerank_model=str(dataset_config["rerank_model"]),
            use_summary=effective_use_summary,
            file_router_top_k=summary_top_k,
            summary_candidate_k=summary_candidate_k,
        )
        sources = pipeline["final_sources"]
        if not sources and retrieval_query != request.query:
            retrieval_query = request.query
            pipeline = build_dataset_retrieval_debug(
                db=db,
                query=retrieval_query,
                dataset_id=request.dataset_id,
                final_k=final_top_k,
                top_k_bm25=top_k_bm25,
                top_k_dense=top_k_dense,
                fusion_method="rrf",
                rerank_enabled=bool(dataset_config["rerank_enabled"]),
                rerank_model=str(dataset_config["rerank_model"]),
                use_summary=effective_use_summary,
                file_router_top_k=summary_top_k,
                summary_candidate_k=summary_candidate_k,
            )
            sources = pipeline["final_sources"]
        retrieval_debug = _to_chat_retrieval_debug(
            dataset_id=request.dataset_id,
            original_query=request.query,
            rewritten_query=retrieval_query if retrieval_query != request.query else None,
            used_query=retrieval_query,
            config={
                "top_k_bm25": top_k_bm25,
                "top_k_dense": top_k_dense,
                "final_top_k": final_top_k,
                "fusion_method": "rrf",
                "use_summary": effective_use_summary,
                "use_summary_requested": requested_use_summary,
                "dataset_has_summary": dataset_has_summary,
                "summarization_mode": str(dataset_config["summarization_mode"]),
                "file_router_top_k": summary_top_k,
                "summary_candidate_k": summary_candidate_k,
                "rerank_enabled": bool(dataset_config["rerank_enabled"]),
                "rerank_model": str(dataset_config["rerank_model"]),
            },
            pipeline=pipeline,
        )
    else:
        sources = retrieve_similar_chunks(
            db=db,
            query=request.query,
            k=final_top_k,
            dataset_id=None,
            top_k_bm25=top_k_bm25,
            top_k_dense=top_k_dense,
            fusion_method="rrf",
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
                    "retrieval_debug": (
                        retrieval_debug.model_dump() if retrieval_debug is not None else None
                    ),
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
    summary_text = ""
    if bool(dataset_config["use_summary"]):
        summary_text = summarize_document(
            text=text,
            mode=str(dataset_config["summarization_mode"]),
            model=settings.chat_model,
        )
        data_file.file_metadata = {
            **(data_file.file_metadata or {}),
            "summary": summary_text,
            "summarization_mode": str(dataset_config["summarization_mode"]),
        }

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
                "file_id": chunk.file_id,
                "content": chunk.content,
                "metadata": chunk.chunk_metadata,
            }
            for chunk in chunk_rows
        ],
    )
    if bool(dataset_config["use_summary"]) and summary_text.strip():
        index_file_summary(
            dataset_id=dataset.id,
            file_id=data_file.id,
            filename=data_file.filename,
            summary=summary_text,
            metadata=data_file.file_metadata,
        )
        summary_embedding = embed_texts([summary_text])[0]
        add_file_summary_embedding(
            dataset_id=dataset.id,
            file_id=data_file.id,
            embedding=summary_embedding,
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


@router.post("/datasets/{dataset_id}/retrieval-debug", response_model=RetrievalDebugResponse)
def debug_dataset_retrieval(
    dataset_id: int,
    request: RetrievalDebugRequest,
    db: Session = Depends(get_db),
) -> RetrievalDebugResponse:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    config = resolve_dataset_config(dataset.config)
    rewritten_query: str | None = None
    if request.enable_query_rewrite:
        rewritten_query = rewrite_query(request.query, model=request.model)
    used_query = rewritten_query or request.query

    final_top_k = request.final_top_k or request.top_k or settings.top_k
    top_k_bm25 = request.top_k_bm25 or settings.top_k
    top_k_dense = request.top_k_dense or settings.top_k
    top_k_bm25 = max(final_top_k, top_k_bm25)
    top_k_dense = max(final_top_k, top_k_dense)
    dataset_has_summary = _dataset_has_summary(db=db, dataset_id=dataset_id)
    requested_use_summary = (
        bool(request.use_summary)
        if request.use_summary is not None
        else bool(config["use_summary"])
    )
    effective_use_summary = requested_use_summary and dataset_has_summary
    summary_top_k = max(1, request.summary_top_k or int(config["file_router_top_k"]))
    summary_candidate_k = max(
        summary_top_k,
        request.summary_candidate_k or max(top_k_bm25, top_k_dense),
    )
    pipeline = build_dataset_retrieval_debug(
        db=db,
        query=used_query,
        dataset_id=dataset_id,
        final_k=final_top_k,
        top_k_bm25=top_k_bm25,
        top_k_dense=top_k_dense,
        fusion_method="rrf",
        rerank_enabled=bool(config["rerank_enabled"]),
        rerank_model=str(config["rerank_model"]),
        use_summary=effective_use_summary,
        file_router_top_k=summary_top_k,
        summary_candidate_k=summary_candidate_k,
    )

    debug_config = {
        **config,
        "use_summary": effective_use_summary,
        "use_summary_requested": requested_use_summary,
        "dataset_has_summary": dataset_has_summary,
        "file_router_top_k": summary_top_k,
        "summary_candidate_k": summary_candidate_k,
    }
    return RetrievalDebugResponse(
        dataset_id=dataset_id,
        original_query=request.query,
        rewritten_query=rewritten_query,
        used_query=used_query,
        config=debug_config,
        routed_file_ids=[int(file_id) for file_id in pipeline.get("file_hits", [])],
        summary_bm25_hits=[
            RetrievalFileStageHit(rank=idx, file_id=hit.file_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline.get("summary_bm25_hits", []), start=1)
        ],
        summary_dense_hits=[
            RetrievalFileStageHit(rank=idx, file_id=hit.file_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline.get("summary_dense_hits", []), start=1)
        ],
        summary_fused_hits=[
            RetrievalFileStageHit(rank=idx, file_id=file_id, score=float(score))
            for idx, (file_id, score) in enumerate(pipeline.get("summary_fused_hits", []), start=1)
        ],
        bm25_hits=[
            RetrievalStageHit(rank=idx, chunk_id=hit.chunk_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline["bm25_hits"], start=1)
        ],
        dense_hits=[
            RetrievalStageHit(rank=idx, chunk_id=hit.chunk_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline["dense_hits"], start=1)
        ],
        fused_hits=[
            RetrievalStageHit(rank=idx, chunk_id=chunk_id, score=float(score))
            for idx, (chunk_id, score) in enumerate(pipeline["fused_hits"], start=1)
        ],
        reranked_hits=[
            RetrievalStageHit(rank=idx, chunk_id=src.chunk_id, score=float(src.score))
            for idx, src in enumerate(pipeline["reranked_sources"], start=1)
        ],
        final_sources=pipeline["final_sources"],
    )


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


def _dataset_has_summary(db: Session, dataset_id: int) -> bool:
    files = db.query(DataFile.file_metadata).filter(DataFile.dataset_id == dataset_id).all()
    for (metadata,) in files:
        payload = metadata or {}
        summary = payload.get("summary") if isinstance(payload, dict) else None
        if isinstance(summary, str) and summary.strip():
            return True
    return False


def _to_chat_retrieval_debug(
    dataset_id: int,
    original_query: str,
    rewritten_query: str | None,
    used_query: str,
    config: dict,
    pipeline: dict,
) -> ChatRetrievalDebug:
    return ChatRetrievalDebug(
        dataset_id=dataset_id,
        original_query=original_query,
        rewritten_query=rewritten_query,
        used_query=used_query,
        config=config,
        routed_file_ids=[int(file_id) for file_id in pipeline.get("file_hits", [])],
        summary_bm25_hits=[
            RetrievalFileStageHit(rank=idx, file_id=hit.file_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline.get("summary_bm25_hits", []), start=1)
        ],
        summary_dense_hits=[
            RetrievalFileStageHit(rank=idx, file_id=hit.file_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline.get("summary_dense_hits", []), start=1)
        ],
        summary_fused_hits=[
            RetrievalFileStageHit(rank=idx, file_id=file_id, score=float(score))
            for idx, (file_id, score) in enumerate(pipeline.get("summary_fused_hits", []), start=1)
        ],
        bm25_hits=[
            RetrievalStageHit(rank=idx, chunk_id=hit.chunk_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline["bm25_hits"], start=1)
        ],
        dense_hits=[
            RetrievalStageHit(rank=idx, chunk_id=hit.chunk_id, score=float(hit.score))
            for idx, hit in enumerate(pipeline["dense_hits"], start=1)
        ],
        fused_hits=[
            RetrievalStageHit(rank=idx, chunk_id=chunk_id, score=float(score))
            for idx, (chunk_id, score) in enumerate(pipeline["fused_hits"], start=1)
        ],
        reranked_hits=[
            RetrievalStageHit(rank=idx, chunk_id=src.chunk_id, score=float(src.score))
            for idx, src in enumerate(pipeline["reranked_sources"], start=1)
        ],
        final_sources=pipeline["final_sources"],
    )
