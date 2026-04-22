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


class WorkflowNode(BaseModel):
    id: str
    type: str
    data: dict = {}
    position: dict | None = None


class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str


class WorkflowRunRequest(BaseModel):
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]


class WorkflowRunResponse(BaseModel):
    output: str
    node_outputs: dict[str, dict]
