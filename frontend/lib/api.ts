export type SourceChunk = {
  document_id: number;
  document_name: string;
  chunk_id: number;
  content: string;
  metadata: Record<string, unknown>;
  score: number;
};

export type ChatResponse = {
  answer: string;
  sources: SourceChunk[];
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

export async function sendChat(query: string, k = 5): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!response.ok) {
    const payload = await safeJson(response);
    throw new Error(payload?.detail ?? "Chat request failed");
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

async function safeJson(response: Response): Promise<any> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}
