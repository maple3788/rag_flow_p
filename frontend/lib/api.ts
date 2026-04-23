export type SourceChunk = {
  dataset_id: number;
  file_id: number;
  filename: string;
  chunk_id: number;
  content: string;
  metadata: Record<string, unknown>;
  score: number;
};

export type ChatResponse = {
  conversation_id: string;
  rewritten_query?: string | null;
  answer: string;
  sources: SourceChunk[];
  evaluation?: EvaluationScores;
  retrieval_debug?: RetrievalDebugResponse | null;
};

export type LlmModel = "qwen3:8b" | "llama3.2:latest";

export type EvaluationScores = {
  faithfulness: number;
  relevance: number;
  context_precision: number;
  rationale?: string | null;
};

export type EvaluateResponse = {
  scores: EvaluationScores;
};

export type EvaluationHistoryItem = {
  id: number;
  conversation_id: string;
  query: string;
  answer: string;
  scores: EvaluationScores;
  created_at: string;
};

export type EvaluationSummaryPoint = {
  period: string;
  faithfulness: number;
  relevance: number;
  context_precision: number;
  count: number;
};

export type WorkflowNode = {
  id: string;
  type: "InputNode" | "RetrieverNode" | "LLMNode" | "AgentNode" | "OutputNode";
  position: { x: number; y: number };
  data: Record<string, unknown>;
};

export type WorkflowEdge = {
  id: string;
  source: string;
  target: string;
};

export type WorkflowRunResponse = {
  output: string;
  node_outputs: Record<string, Record<string, unknown>>;
};

export type Dataset = {
  id: number;
  name: string;
  description: string;
  config: Record<string, unknown>;
  created_at: string;
};

export type DatasetFile = {
  id: number;
  dataset_id: number;
  filename: string;
  raw_text: string;
  metadata: Record<string, unknown>;
};

export type DatasetChunk = {
  id: number;
  file_id: number;
  dataset_id: number;
  content: string;
  metadata: Record<string, unknown>;
};

export type RetrievalStageHit = {
  rank: number;
  chunk_id: number;
  score: number;
};

export type RetrievalDebugResponse = {
  dataset_id: number;
  original_query: string;
  rewritten_query?: string | null;
  used_query: string;
  config: Record<string, unknown>;
  bm25_hits: RetrievalStageHit[];
  dense_hits: RetrievalStageHit[];
  fused_hits: RetrievalStageHit[];
  reranked_hits: RetrievalStageHit[];
  final_sources: SourceChunk[];
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

export async function uploadFile(file: File): Promise<{
  document_id: number;
  file_name: string;
  chunks_indexed: number;
}> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Upload failed");
  }
  return response.json();
}

export async function sendChat(
  query: string,
  k: number | undefined = undefined,
  model: LlmModel = "qwen3:8b",
  conversationId?: string,
  options?: {
    enableQueryRewrite?: boolean;
    enableRerank?: boolean;
    datasetId?: number;
    topKBm25?: number;
    topKDense?: number;
    finalTopK?: number;
  }
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      k,
      model,
      conversation_id: conversationId,
      dataset_id: options?.datasetId,
      top_k_bm25: options?.topKBm25,
      top_k_dense: options?.topKDense,
      final_top_k: options?.finalTopK,
      enable_query_rewrite: Boolean(options?.enableQueryRewrite),
      enable_rerank: Boolean(options?.enableRerank),
    }),
  });
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Chat request failed");
  }
  return response.json();
}

export async function streamChat(
  query: string,
  k: number | undefined,
  model: LlmModel,
  conversationId: string | undefined,
  options: {
    enableQueryRewrite?: boolean;
    enableRerank?: boolean;
    datasetId?: number;
    topKBm25?: number;
    topKDense?: number;
    finalTopK?: number;
  },
  handlers: {
    onToken: (token: string) => void;
    onDone: (payload: ChatResponse) => void;
  }
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      k,
      model,
      conversation_id: conversationId,
      dataset_id: options.datasetId,
      top_k_bm25: options.topKBm25,
      top_k_dense: options.topKDense,
      final_top_k: options.finalTopK,
      enable_query_rewrite: Boolean(options.enableQueryRewrite),
      enable_rerank: Boolean(options.enableRerank),
    }),
  });
  if (!response.ok || !response.body) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Chat stream failed");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      if (!line.trim()) continue;
      let event: any;
      try {
        event = JSON.parse(line);
      } catch {
        continue;
      }
      if (event.type === "token") {
        handlers.onToken(String(event.content ?? ""));
      }
      if (event.type === "done") {
        handlers.onDone({
          conversation_id: String(event.conversation_id ?? ""),
          rewritten_query: event.rewritten_query ?? null,
          answer: String(event.answer ?? ""),
          sources: Array.isArray(event.sources) ? event.sources : [],
          evaluation: event.evaluation,
          retrieval_debug: event.retrieval_debug ?? null,
        });
      }
    }
  }
}

export async function getEvaluationHistory(
  limit = 20,
  conversationId?: string
): Promise<EvaluationHistoryItem[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (conversationId) params.set("conversation_id", conversationId);
  const response = await fetch(`${API_BASE_URL}/evaluations?${params.toString()}`);
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Failed to load evaluation history");
  }
  return response.json();
}

export async function getEvaluationSummary(
  limit = 200,
  conversationId?: string
): Promise<EvaluationSummaryPoint[]> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (conversationId) params.set("conversation_id", conversationId);
  const response = await fetch(`${API_BASE_URL}/evaluations/summary?${params.toString()}`);
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Failed to load evaluation summary");
  }
  return response.json();
}

export async function runWorkflow(
  nodes: WorkflowNode[],
  edges: WorkflowEdge[]
): Promise<WorkflowRunResponse> {
  const response = await fetch(`${API_BASE_URL}/workflow/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ nodes, edges }),
  });
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Workflow run failed");
  }
  return response.json();
}

export async function evaluateAnswer(
  query: string,
  answer: string,
  contexts: string[],
  model: LlmModel = "qwen3:8b"
): Promise<EvaluateResponse> {
  const response = await fetch(`${API_BASE_URL}/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, answer, contexts, model }),
  });
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Evaluation failed");
  }
  return response.json();
}

export async function createDataset(payload: {
  name: string;
  description?: string;
  config?: Record<string, unknown>;
}): Promise<Dataset> {
  const response = await fetch(`${API_BASE_URL}/datasets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: payload.name,
      description: payload.description ?? "",
      config: payload.config ?? {},
    }),
  });
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to create dataset");
  }
  return response.json();
}

export async function listDatasets(): Promise<Dataset[]> {
  const response = await fetch(`${API_BASE_URL}/datasets`);
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to list datasets");
  }
  return response.json();
}

export async function getDataset(datasetId: number): Promise<Dataset> {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`);
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to get dataset");
  }
  return response.json();
}

export async function updateDataset(
  datasetId: number,
  payload: { description?: string; config?: Record<string, unknown> }
): Promise<Dataset> {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to update dataset");
  }
  return response.json();
}

export async function deleteDataset(datasetId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to delete dataset");
  }
}

export async function uploadDatasetFile(datasetId: number, file: File): Promise<DatasetFile> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/files`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to upload dataset file");
  }
  return response.json();
}

export async function listDatasetFiles(datasetId: number): Promise<DatasetFile[]> {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/files`);
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to list files");
  }
  return response.json();
}

export async function deleteDatasetFile(datasetId: number, fileId: number): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/files/${fileId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to delete file");
  }
}

export async function listDatasetChunks(
  datasetId: number,
  fileId?: number
): Promise<DatasetChunk[]> {
  const params = new URLSearchParams();
  if (typeof fileId === "number") params.set("file_id", String(fileId));
  const query = params.toString();
  const response = await fetch(
    `${API_BASE_URL}/datasets/${datasetId}/chunks${query ? `?${query}` : ""}`
  );
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to list chunks");
  }
  return response.json();
}

export async function debugDatasetRetrieval(
  datasetId: number,
  payload: {
    query: string;
    top_k?: number;
    top_k_bm25?: number;
    top_k_dense?: number;
    final_top_k?: number;
    enable_query_rewrite?: boolean;
    model?: string;
  }
): Promise<RetrievalDebugResponse> {
  const response = await fetch(`${API_BASE_URL}/datasets/${datasetId}/retrieval-debug`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await safeJson(response);
    throw new Error(body?.detail ?? "Failed to run retrieval debug");
  }
  return response.json();
}

async function safeJson(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}
