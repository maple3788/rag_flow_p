export type SourceChunk = {
  document_id: number;
  document_name: string;
  chunk_id: number;
  content: string;
  metadata: Record<string, unknown>;
  score: number;
};

export type ChatResponse = {
  conversation_id: string;
  answer: string;
  sources: SourceChunk[];
  evaluation?: EvaluationScores;
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
  type: "InputNode" | "RetrieverNode" | "LLMNode" | "OutputNode";
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
  k = 5,
  model: LlmModel = "qwen3:8b",
  conversationId?: string
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k, model, conversation_id: conversationId }),
  });
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Chat request failed");
  }
  return response.json();
}

export async function streamChat(
  query: string,
  k: number,
  model: LlmModel,
  conversationId: string | undefined,
  handlers: {
    onToken: (token: string) => void;
    onDone: (payload: ChatResponse) => void;
  }
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k, model, conversation_id: conversationId }),
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
          answer: String(event.answer ?? ""),
          sources: Array.isArray(event.sources) ? event.sources : [],
          evaluation: event.evaluation,
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

async function safeJson(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}
