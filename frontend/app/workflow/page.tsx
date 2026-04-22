"use client";

import { useCallback, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";

import { runWorkflow, type WorkflowEdge, type WorkflowNode } from "@/lib/api";

type NodeTypeName = "InputNode" | "RetrieverNode" | "LLMNode" | "AgentNode" | "OutputNode";

const defaultNodes: Node[] = [
  {
    id: "input-1",
    type: "default",
    position: { x: 40, y: 100 },
    data: { label: "InputNode", query: "What is in my documents?" },
  },
  {
    id: "retriever-1",
    type: "default",
    position: { x: 320, y: 100 },
    data: { label: "RetrieverNode", k: 5 },
  },
  {
    id: "llm-1",
    type: "default",
    position: { x: 600, y: 100 },
    data: { label: "LLMNode" },
  },
  {
    id: "output-1",
    type: "default",
    position: { x: 860, y: 100 },
    data: { label: "OutputNode" },
  },
];

const defaultEdges: Edge[] = [
  { id: "e-input-retriever", source: "input-1", target: "retriever-1" },
  { id: "e-retriever-llm", source: "retriever-1", target: "llm-1" },
  { id: "e-llm-output", source: "llm-1", target: "output-1" },
];

export default function WorkflowPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState(defaultNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(defaultEdges);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [jsonGraph, setJsonGraph] = useState("");
  const [running, setRunning] = useState(false);

  const onConnect = useCallback(
    (params: Edge | Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const counts = useMemo(
    () => ({
      InputNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "InputNode").length,
      RetrieverNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "RetrieverNode")
        .length,
      LLMNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "LLMNode").length,
      AgentNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "AgentNode").length,
      OutputNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "OutputNode").length,
    }),
    [nodes]
  );
  const selectedNodeIds = useMemo(
    () => nodes.filter((node) => node.selected).map((node) => node.id),
    [nodes]
  );
  const selectedEdgeIds = useMemo(
    () => edges.filter((edge) => edge.selected).map((edge) => edge.id),
    [edges]
  );
  const selectedNode = useMemo(
    () => nodes.find((node) => node.selected) ?? null,
    [nodes]
  );

  function addNode(type: NodeTypeName) {
    const id = `${type.toLowerCase()}-${Date.now()}`;
    const defaultData: Record<string, unknown> = { label: type };
    if (type === "InputNode") defaultData.query = "Ask your question here";
    if (type === "RetrieverNode") defaultData.k = 5;
    if (type === "LLMNode")
      defaultData.template = "Use retrieved context to answer the user question.";
    if (type === "AgentNode") {
      defaultData.max_steps = 5;
      defaultData.k = 5;
      defaultData.use_web_search = false;
      defaultData.model = "qwen3:8b";
    }

    setNodes((prev) => [
      ...prev,
      {
        id,
        type: "default",
        position: { x: 100 + prev.length * 30, y: 260 + prev.length * 20 },
        data: defaultData,
      },
    ]);
  }

  function exportGraph() {
    const payload = {
      nodes: serializeNodes(nodes),
      edges: serializeEdges(edges),
    };
    setJsonGraph(JSON.stringify(payload, null, 2));
  }

  function importGraph() {
    try {
      const parsed = JSON.parse(jsonGraph) as { nodes: WorkflowNode[]; edges: WorkflowEdge[] };
      const nextNodes: Node[] = parsed.nodes.map((n) => ({
        id: n.id,
        type: "default",
        position: n.position,
        data: n.data,
      }));
      const nextEdges: Edge[] = parsed.edges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
      }));
      setNodes(nextNodes);
      setEdges(nextEdges);
      setError("");
    } catch {
      setError("Invalid workflow JSON");
    }
  }

  async function run() {
    setRunning(true);
    setError("");
    setResult("");
    try {
      const response = await runWorkflow(serializeNodes(nodes), serializeEdges(edges));
      setResult(response.output);
      setJsonGraph(JSON.stringify(response.node_outputs, null, 2));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run workflow");
    } finally {
      setRunning(false);
    }
  }

  function deleteSelected() {
    if (selectedNodeIds.length === 0 && selectedEdgeIds.length === 0) return;

    const removedNodeSet = new Set(selectedNodeIds);
    const removedEdgeSet = new Set(selectedEdgeIds);

    setNodes((prev) => prev.filter((node) => !removedNodeSet.has(node.id)));
    setEdges((prev) =>
      prev.filter(
        (edge) =>
          !removedEdgeSet.has(edge.id) &&
          !removedNodeSet.has(edge.source) &&
          !removedNodeSet.has(edge.target)
      )
    );
  }

  function updateSelectedNodeData(patch: Record<string, unknown>) {
    if (!selectedNode) return;
    setNodes((prev) =>
      prev.map((node) =>
        node.id === selectedNode.id
          ? { ...node, data: { ...(node.data ?? {}), ...patch } }
          : node
      )
    );
  }

  return (
    <section className="workflow-wrap">
      <h2>Workflow Builder</h2>
      <p className="muted">
        Build node-based RAG pipelines with Input → Retriever → LLM → Output.
      </p>

      <div className="workflow-toolbar">
        <button className="button secondary" onClick={() => addNode("InputNode")}>
          Add InputNode
        </button>
        <button className="button secondary" onClick={() => addNode("RetrieverNode")}>
          Add RetrieverNode
        </button>
        <button className="button secondary" onClick={() => addNode("LLMNode")}>
          Add LLMNode
        </button>
        <button className="button secondary" onClick={() => addNode("AgentNode")}>
          Add AgentNode
        </button>
        <button className="button secondary" onClick={() => addNode("OutputNode")}>
          Add OutputNode
        </button>
        <button
          className="button secondary"
          onClick={deleteSelected}
          disabled={selectedNodeIds.length === 0 && selectedEdgeIds.length === 0}
        >
          Delete Selected
        </button>
        <button className="button" onClick={run} disabled={running}>
          {running ? "Running..." : "Run Workflow"}
        </button>
      </div>

      <div className="workflow-stats muted">
        Nodes: Input {counts.InputNode} | Retriever {counts.RetrieverNode} | LLM {counts.LLMNode} |
        Agent {counts.AgentNode} | Output {counts.OutputNode}
      </div>
      <div className="workflow-stats muted">
        Selected: {selectedNodeIds.length} node(s), {selectedEdgeIds.length} edge(s). Use Delete
        Selected button or keyboard Delete/Backspace.
      </div>

      <div className="flow-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          fitView
          deleteKeyCode={["Delete", "Backspace"]}
        >
          <MiniMap />
          <Controls />
          <Background />
        </ReactFlow>
      </div>

      <div className="workflow-inspector">
        <h3>Node Inspector</h3>
        {!selectedNode && <p className="muted">Select a node to edit its settings.</p>}
        {selectedNode && (
          <div className="inspector-fields">
            <p className="muted">
              Editing: <strong>{String(selectedNode.data?.label ?? selectedNode.id)}</strong>
            </p>
            <p className="muted">Node ID: {selectedNode.id}</p>

            {nodeTypeFromLabel(selectedNode.data?.label) === "InputNode" && (
              <label>
                Query
                <textarea
                  className="json-box"
                  value={String(selectedNode.data?.query ?? "")}
                  onChange={(event) =>
                    updateSelectedNodeData({ query: event.target.value })
                  }
                />
              </label>
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "RetrieverNode" && (
              <label>
                Top K
                <input
                  className="inspector-input"
                  type="number"
                  min={1}
                  max={20}
                  value={Number(selectedNode.data?.k ?? 5)}
                  onChange={(event) =>
                    updateSelectedNodeData({
                      k: Number(event.target.value || 5),
                    })
                  }
                />
              </label>
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "LLMNode" && (
              <>
                <label>
                  Prompt Template
                  <textarea
                    className="json-box"
                    value={String(selectedNode.data?.template ?? "")}
                    onChange={(event) =>
                      updateSelectedNodeData({ template: event.target.value })
                    }
                  />
                </label>
                <label>
                  Model
                  <select
                    className="inspector-input"
                    value={String(selectedNode.data?.model ?? "qwen3:8b")}
                    onChange={(event) =>
                      updateSelectedNodeData({ model: event.target.value })
                    }
                  >
                    <option value="qwen3:8b">qwen3:8b</option>
                    <option value="llama3.2:latest">llama3.2:latest</option>
                  </select>
                </label>
              </>
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "AgentNode" && (
              <>
                <label>
                  Query (optional)
                  <textarea
                    className="json-box"
                    value={String(selectedNode.data?.query ?? "")}
                    onChange={(event) =>
                      updateSelectedNodeData({ query: event.target.value })
                    }
                  />
                </label>
                <label>
                  Top K
                  <input
                    className="inspector-input"
                    type="number"
                    min={1}
                    max={20}
                    value={Number(selectedNode.data?.k ?? 5)}
                    onChange={(event) =>
                      updateSelectedNodeData({
                        k: Number(event.target.value || 5),
                      })
                    }
                  />
                </label>
                <label>
                  Max Steps
                  <input
                    className="inspector-input"
                    type="number"
                    min={1}
                    max={20}
                    value={Number(selectedNode.data?.max_steps ?? 5)}
                    onChange={(event) =>
                      updateSelectedNodeData({
                        max_steps: Number(event.target.value || 5),
                      })
                    }
                  />
                </label>
                <label>
                  Model
                  <select
                    className="inspector-input"
                    value={String(selectedNode.data?.model ?? "qwen3:8b")}
                    onChange={(event) =>
                      updateSelectedNodeData({ model: event.target.value })
                    }
                  >
                    <option value="qwen3:8b">qwen3:8b</option>
                    <option value="llama3.2:latest">llama3.2:latest</option>
                  </select>
                </label>
                <label className="inspector-checkbox">
                  <input
                    type="checkbox"
                    checked={Boolean(selectedNode.data?.use_web_search ?? false)}
                    onChange={(event) =>
                      updateSelectedNodeData({ use_web_search: event.target.checked })
                    }
                  />
                  Enable web search tool
                </label>
              </>
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "OutputNode" && (
              <p className="muted">OutputNode has no editable fields.</p>
            )}
          </div>
        )}
      </div>

      <div className="workflow-toolbar">
        <button className="button secondary" onClick={exportGraph}>
          Export Graph JSON
        </button>
        <button className="button secondary" onClick={importGraph}>
          Load Graph JSON
        </button>
      </div>

      <textarea
        className="json-box"
        value={jsonGraph}
        onChange={(event) => setJsonGraph(event.target.value)}
        placeholder='{"nodes":[...],"edges":[...]}'
      />

      {result && (
        <div className="card">
          <h3>Workflow Output</h3>
          <p>{result}</p>
        </div>
      )}

      {error && <p className="error">{error}</p>}
    </section>
  );
}

function nodeTypeFromLabel(label: unknown): NodeTypeName | null {
  if (
    label === "InputNode" ||
    label === "RetrieverNode" ||
    label === "LLMNode" ||
    label === "AgentNode" ||
    label === "OutputNode"
  ) {
    return label;
  }
  return null;
}

function serializeNodes(nodes: Node[]): WorkflowNode[] {
  return nodes
    .map((node) => {
      const nodeType = nodeTypeFromLabel(node.data?.label);
      if (!nodeType) return null;
      return {
        id: node.id,
        type: nodeType,
        position: node.position,
        data: normalizeNodeData(node.data ?? {}, nodeType),
      };
    })
    .filter((item): item is WorkflowNode => item !== null);
}

function serializeEdges(edges: Edge[]): WorkflowEdge[] {
  return edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
  }));
}

function normalizeNodeData(data: Record<string, unknown>, type: NodeTypeName): Record<string, unknown> {
  if (type === "InputNode") {
    return { query: String(data.query ?? "") };
  }
  if (type === "RetrieverNode") {
    return { k: Number(data.k ?? 5) };
  }
  if (type === "LLMNode") {
    return { template: String(data.template ?? "") };
  }
  if (type === "AgentNode") {
    return {
      query: String(data.query ?? ""),
      k: Number(data.k ?? 5),
      max_steps: Number(data.max_steps ?? 5),
      use_web_search: Boolean(data.use_web_search ?? false),
      model: String(data.model ?? "qwen3:8b"),
    };
  }
  return {};
}
