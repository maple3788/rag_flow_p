from datetime import datetime

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: int
    file_name: str
    chunks_indexed: int


class SourceChunk(BaseModel):
    dataset_id: int
    file_id: int
    filename: str
    chunk_id: int
    content: str
    metadata: dict
    score: float


class ChatRequest(BaseModel):
    query: str
    k: int | None = None
    top_k_bm25: int | None = None
    top_k_dense: int | None = None
    final_top_k: int | None = None
    dataset_id: int | None = None
    model: str | None = None
    conversation_id: str | None = None
    enable_query_rewrite: bool = False
    enable_rerank: bool = False


class EvaluationScores(BaseModel):
    faithfulness: float
    relevance: float
    context_precision: float
    rationale: str | None = None


class RetrievalStageHit(BaseModel):
    rank: int
    chunk_id: int
    score: float


class ChatRetrievalDebug(BaseModel):
    dataset_id: int | None = None
    original_query: str
    rewritten_query: str | None = None
    used_query: str
    config: dict
    bm25_hits: list[RetrievalStageHit]
    dense_hits: list[RetrievalStageHit]
    fused_hits: list[RetrievalStageHit]
    reranked_hits: list[RetrievalStageHit]
    final_sources: list[SourceChunk]


class ChatResponse(BaseModel):
    conversation_id: str
    rewritten_query: str | None = None
    answer: str
    sources: list[SourceChunk]
    evaluation: EvaluationScores | None = None
    retrieval_debug: ChatRetrievalDebug | None = None


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


class DatasetCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str = ""
    config: dict = Field(default_factory=dict)


class DatasetUpdateRequest(BaseModel):
    description: str | None = None
    config: dict | None = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: str
    config: dict
    created_at: datetime


class DatasetFileResponse(BaseModel):
    id: int
    dataset_id: int
    filename: str
    raw_text: str
    metadata: dict


class DatasetChunkResponse(BaseModel):
    id: int
    file_id: int
    dataset_id: int
    content: str
    metadata: dict


class RetrievalDebugRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None
    top_k_bm25: int | None = None
    top_k_dense: int | None = None
    final_top_k: int | None = None
    enable_query_rewrite: bool = True
    model: str | None = None


class RetrievalDebugResponse(BaseModel):
    dataset_id: int
    original_query: str
    rewritten_query: str | None = None
    used_query: str
    config: dict
    bm25_hits: list[RetrievalStageHit]
    dense_hits: list[RetrievalStageHit]
    fused_hits: list[RetrievalStageHit]
    reranked_hits: list[RetrievalStageHit]
    final_sources: list[SourceChunk]
