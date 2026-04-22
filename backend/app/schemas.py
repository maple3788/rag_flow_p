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
    model: str | None = None
    conversation_id: str | None = None
    enable_query_rewrite: bool = False
    enable_rerank: bool = False


class EvaluationScores(BaseModel):
    faithfulness: float
    relevance: float
    context_precision: float
    rationale: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    rewritten_query: str | None = None
    answer: str
    sources: list[SourceChunk]
    evaluation: EvaluationScores | None = None


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


class EvaluateRequest(BaseModel):
    query: str
    answer: str
    contexts: list[str]
    model: str | None = None
    conversation_id: str | None = None


class EvaluateResponse(BaseModel):
    scores: EvaluationScores


class EvaluationHistoryItem(BaseModel):
    id: int
    conversation_id: str
    query: str
    answer: str
    scores: EvaluationScores
    created_at: str


class EvaluationSummaryPoint(BaseModel):
    period: str
    faithfulness: float
    relevance: float
    context_precision: float
    count: int
