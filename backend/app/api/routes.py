from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import Chunk, Document
from app.schemas import ChatRequest, ChatResponse, UploadResponse
from app.services.chat import generate_answer
from app.services.document_parser import parse_uploaded_file
from app.services.embeddings import embed_texts
from app.services.retrieval import retrieve_similar_chunks
from app.services.text_splitter import split_text_recursive

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

    k = request.k or settings.top_k
    sources = retrieve_similar_chunks(db=db, query=request.query, k=k)
    answer = generate_answer(query=request.query, sources=sources)
    return ChatResponse(answer=answer, sources=sources)
