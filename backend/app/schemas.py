from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: int
    file_name: str
    chunks_indexed: int


class SourceChunk(BaseModel):
    document_id: int
    document_name: str
    chunk_id: int
    content: str
    metadata: dict
    score: float


class ChatRequest(BaseModel):
    query: str
    k: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
