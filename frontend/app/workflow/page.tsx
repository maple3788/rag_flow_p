"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
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

import {
  listDatasets,
  runWorkflowStream,
  type Dataset,
  type WorkflowEdge,
  type WorkflowNode,
  type WorkflowRunStreamEvent,
} from "@/lib/api";

type NodeTypeName =
  | "InputNode"
  | "RetrieverNode"
  | "LLMNode"
  | "AgentNode"
  | "PlannerNode"
  | "ToolSelectorNode"
  | "ToolExecutorNode"
  | "ReflectionNode"
  | "FinalAnswerNode"
  | "OutputNode";

const defaultNodes: Node[] = [
  {
    id: "input-1",
    type: "default",
    position: { x: 80, y: 80 },
    data: { label: "InputNode", query: "Ask your question here" },
  },
  {
    id: "planner-1",
    type: "default",
    position: { x: 320, y: 80 },
    data: { label: "PlannerNode", model: "qwen3:8b" },
  },
  {
    id: "selector-1",
    type: "default",
    position: { x: 560, y: 80 },
    data: { label: "ToolSelectorNode", strategy: "plan_first" },
  },
  {
    id: "executor-1",
    type: "default",
    position: { x: 820, y: 80 },
    data: { label: "ToolExecutorNode", k: 5, model: "qwen3:8b", dataset_id: "" },
  },
  {
    id: "reflection-1",
    type: "default",
    position: { x: 560, y: 240 },
    data: { label: "ReflectionNode", max_loops: 3, model: "qwen3:8b" },
  },
  {
    id: "final-1",
    type: "default",
    position: { x: 820, y: 240 },
    data: { label: "FinalAnswerNode", model: "qwen3:8b" },
  },
  {
    id: "output-1",
    type: "default",
    position: { x: 1060, y: 240 },
    data: { label: "OutputNode" },
  },
];

const defaultEdges: Edge[] = [
  { id: "e-input-planner", source: "input-1", target: "planner-1" },
  { id: "e-planner-selector", source: "planner-1", target: "selector-1" },
  {
    id: "e-selector-executor",
    source: "selector-1",
    target: "executor-1",
    label: "tool!=finish",
    ...( { condition: "tool!=finish" } as any ),
  },
  { id: "e-executor-reflection", source: "executor-1", target: "reflection-1" },
  {
    id: "e-reflection-planner",
    source: "reflection-1",
    target: "planner-1",
    label: "continue=true",
  },
  {
    id: "e-reflection-final",
    source: "reflection-1",
    target: "final-1",
    label: "continue=false",
  },
  { id: "e-final-output", source: "final-1", target: "output-1" },
];

export default function WorkflowPage() {
  const [nodes, setNodes, onNodesChange] = useNodesState(defaultNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(defaultEdges);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");
  const [jsonGraph, setJsonGraph] = useState("");
  const [running, setRunning] = useState(false);
  const [activeEdgeIds, setActiveEdgeIds] = useState<string[]>([]);
  const [currentNodeId, setCurrentNodeId] = useState<string | null>(null);
  const [streamEvents, setStreamEvents] = useState<WorkflowRunStreamEvent[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setDatasetsLoading(true);
    setDatasetsError("");
    listDatasets()
      .then((items) => {
        if (cancelled) return;
        setDatasets(items);
      })
      .catch((err) => {
        if (cancelled) return;
        setDatasetsError(err instanceof Error ? err.message : "Failed to load datasets");
      })
      .finally(() => {
        if (cancelled) return;
        setDatasetsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

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
      PlannerNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "PlannerNode").length,
      ToolSelectorNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "ToolSelectorNode")
        .length,
      ToolExecutorNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "ToolExecutorNode")
        .length,
      ReflectionNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "ReflectionNode")
        .length,
      FinalAnswerNode: nodes.filter((n) => nodeTypeFromLabel(n.data?.label) === "FinalAnswerNode")
        .length,
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
  const selectedEdge = useMemo(
    () => edges.find((edge) => edge.selected) ?? null,
    [edges]
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
    if (type === "PlannerNode") defaultData.model = "qwen3:8b";
    if (type === "ToolSelectorNode") defaultData.strategy = "plan_first";
    if (type === "ToolExecutorNode") {
      defaultData.k = 5;
      defaultData.final_k = 5;
      defaultData.top_k_bm25 = 8;
      defaultData.top_k_dense = 8;
      defaultData.rerank_enabled = true;
      defaultData.model = "qwen3:8b";
      defaultData.dataset_id = "";
    }
    if (type === "ReflectionNode") {
      defaultData.model = "qwen3:8b";
      defaultData.max_loops = 3;
    }
    if (type === "FinalAnswerNode") defaultData.model = "qwen3:8b";

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
        label: e.condition ?? undefined,
        ...(e.condition ? ({ condition: e.condition } as any) : {}),
      }));
      setNodes(nextNodes);
      setEdges(nextEdges);
      setActiveEdgeIds([]);
      setError("");
    } catch {
      setError("Invalid workflow JSON");
    }
  }

  async function run() {
    setRunning(true);
    setError("");
    setResult("");
    setStreamEvents([]);
    setCurrentNodeId(null);
    setActiveEdgeIds([]);
    setNodes((prev) => prev.map((node) => ({ ...node, style: {} })));
    try {
      await runWorkflowStream(serializeNodes(nodes), serializeEdges(edges), {
        onEvent: (event) => {
          handleStreamEvent(event);
        },
        onDone: (response) => {
          setResult(response.output);
          setJsonGraph(JSON.stringify(response.node_outputs, null, 2));
          setActiveEdgeIds((response.route_trace ?? []).map((item) => item.edge_id));
          setEdges((prev) =>
            prev.map((edge) =>
              (response.route_trace ?? []).some((item) => item.edge_id === edge.id)
                ? {
                    ...edge,
                    animated: true,
                    style: { ...(edge.style ?? {}), stroke: "#2563eb", strokeWidth: 2.5 },
                  }
                : {
                    ...edge,
                    animated: false,
                    style: { ...(edge.style ?? {}), stroke: "#9ca3af", strokeWidth: 1.2 },
                  }
            )
          );
        },
      });
      setCurrentNodeId(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run workflow");
      setActiveEdgeIds([]);
      setCurrentNodeId(null);
      setEdges((prev) =>
        prev.map((edge) => ({
          ...edge,
          animated: false,
          style: { ...(edge.style ?? {}), stroke: "#9ca3af", strokeWidth: 1.2 },
        }))
      );
    } finally {
      setRunning(false);
    }
  }

  function handleStreamEvent(event: WorkflowRunStreamEvent) {
    setStreamEvents((prev) => [...prev, event]);
    if (event.type === "started") {
      return;
    }
    if (event.type === "node_start") {
      setCurrentNodeId(event.node_id);
      setNodes((prev) =>
        prev.map((node) =>
          node.id === event.node_id
            ? {
                ...node,
                style: { ...(node.style ?? {}), border: "2px solid #f59e0b", boxShadow: "0 0 0 2px rgba(245, 158, 11, 0.2)" },
              }
            : {
                ...node,
                style: { ...(node.style ?? {}), border: "1px solid #3a4250", boxShadow: "none" },
              }
        )
      );
      return;
    }
    if (event.type === "node_complete") {
      setNodes((prev) =>
        prev.map((node) =>
          node.id === event.node_id
            ? {
                ...node,
                style: { ...(node.style ?? {}), border: "2px solid #22c55e", boxShadow: "0 0 0 2px rgba(34, 197, 94, 0.18)" },
              }
            : node
        )
      );
      return;
    }
    if (event.type === "edge_traversed") {
      return;
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

  function updateSelectedEdgeData(patch: Record<string, unknown>) {
    if (!selectedEdge) return;
    setEdges((prev) =>
      prev.map((edge) =>
        edge.id === selectedEdge.id
          ? { ...edge, ...patch }
          : edge
      )
    );
  }

  return (
    <section className="workflow-wrap workflow-wrap-wide">
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
        <button className="button secondary" onClick={() => addNode("PlannerNode")}>
          Add PlannerNode
        </button>
        <button className="button secondary" onClick={() => addNode("ToolSelectorNode")}>
          Add ToolSelectorNode
        </button>
        <button className="button secondary" onClick={() => addNode("ToolExecutorNode")}>
          Add ToolExecutorNode
        </button>
        <button className="button secondary" onClick={() => addNode("ReflectionNode")}>
          Add ReflectionNode
        </button>
        <button className="button secondary" onClick={() => addNode("FinalAnswerNode")}>
          Add FinalAnswerNode
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
        Agent {counts.AgentNode} | Planner {counts.PlannerNode} | Selector {counts.ToolSelectorNode}
        | Executor {counts.ToolExecutorNode} | Reflection {counts.ReflectionNode} | FinalAnswer{" "}
        {counts.FinalAnswerNode} | Output {counts.OutputNode}
      </div>
      <div className="workflow-stats muted">
        Selected: {selectedNodeIds.length} node(s), {selectedEdgeIds.length} edge(s). Use Delete
        Selected button or keyboard Delete/Backspace.
      </div>
      {activeEdgeIds.length > 0 && (
        <div className="workflow-stats muted">
          Executed edges: {activeEdgeIds.length} (animated on canvas)
        </div>
      )}
      {running && (
        <div className="workflow-stats muted">
          Current node: {currentNodeId ?? "starting..."}
        </div>
      )}

      <div className="workflow-main-grid">
        <div className="workflow-left-panel">
          <div className="workflow-left-grid">
            <div className="workflow-inspector workflow-inspector-stream">
              <h3>Run Stream</h3>
              <div className="workflow-log-box">
                {streamEvents.length === 0 && (
                  <p className="muted">No events yet. Click Run Workflow to stream each step.</p>
                )}
                {streamEvents.map((event, idx) => (
                  <details key={idx} className="workflow-event">
                    <summary className="workflow-event-summary">{eventLabel(event)}</summary>
                    <div className="workflow-event-body">
                      <JsonTree data={event} />
                    </div>
                  </details>
                ))}
              </div>
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
          </div>
        </div>

        <aside className="workflow-right-panel">
          <div className="workflow-inspector workflow-inspector-sticky">
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

            {nodeTypeFromLabel(selectedNode.data?.label) === "PlannerNode" && (
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
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "ToolSelectorNode" && (
              <label>
                Strategy
                <input
                  className="inspector-input"
                  value={String(selectedNode.data?.strategy ?? "plan_first")}
                  onChange={(event) =>
                    updateSelectedNodeData({ strategy: event.target.value })
                  }
                />
              </label>
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "ToolExecutorNode" && (
              <>
                <label>
                  Final K
                  <input
                    className="inspector-input"
                    type="number"
                    min={1}
                    max={50}
                    value={Number(selectedNode.data?.final_k ?? selectedNode.data?.k ?? 5)}
                    onChange={(event) =>
                      updateSelectedNodeData({
                        final_k: Number(event.target.value || 5),
                      })
                    }
                  />
                </label>
                <label>
                  Top K BM25
                  <input
                    className="inspector-input"
                    type="number"
                    min={1}
                    max={100}
                    value={Number(selectedNode.data?.top_k_bm25 ?? 8)}
                    onChange={(event) =>
                      updateSelectedNodeData({
                        top_k_bm25: Number(event.target.value || 8),
                      })
                    }
                  />
                </label>
                <label>
                  Top K Dense
                  <input
                    className="inspector-input"
                    type="number"
                    min={1}
                    max={100}
                    value={Number(selectedNode.data?.top_k_dense ?? 8)}
                    onChange={(event) =>
                      updateSelectedNodeData({
                        top_k_dense: Number(event.target.value || 8),
                      })
                    }
                  />
                </label>
                <label className="inspector-checkbox">
                  <input
                    type="checkbox"
                    checked={Boolean(selectedNode.data?.rerank_enabled ?? true)}
                    onChange={(event) =>
                      updateSelectedNodeData({ rerank_enabled: event.target.checked })
                    }
                  />
                  Enable rerank
                </label>
                <label>
                  Dataset (optional)
                  <select
                    className="inspector-input"
                    value={String(selectedNode.data?.dataset_id ?? "")}
                    onChange={(event) =>
                      updateSelectedNodeData({ dataset_id: event.target.value })
                    }
                  >
                    <option value="">All datasets (no filter)</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.id} value={String(dataset.id)}>
                        {dataset.name} (id: {dataset.id})
                      </option>
                    ))}
                  </select>
                </label>
                {datasetsLoading && <p className="muted">Loading datasets...</p>}
                {datasetsError && <p className="error">{datasetsError}</p>}
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

            {nodeTypeFromLabel(selectedNode.data?.label) === "ReflectionNode" && (
              <>
                <label>
                  Max Loops
                  <input
                    className="inspector-input"
                    type="number"
                    min={1}
                    max={20}
                    value={Number(selectedNode.data?.max_loops ?? 3)}
                    onChange={(event) =>
                      updateSelectedNodeData({
                        max_loops: Number(event.target.value || 3),
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
              </>
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "FinalAnswerNode" && (
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
            )}

            {nodeTypeFromLabel(selectedNode.data?.label) === "OutputNode" && (
              <p className="muted">OutputNode has no editable fields.</p>
            )}
          </div>
        )}
          </div>
        </aside>
      </div>

      <div className="workflow-inspector">
        <h3>Edge Inspector</h3>
        {!selectedEdge && <p className="muted">Select an edge to edit routing condition.</p>}
        {selectedEdge && (
          <div className="inspector-fields">
            <p className="muted">
              Editing edge: <strong>{selectedEdge.id}</strong>
            </p>
            <p className="muted">
              {selectedEdge.source} -&gt; {selectedEdge.target}
            </p>
            <label>
              Condition (optional)
              <input
                className="inspector-input"
                value={String((selectedEdge as any).condition ?? "")}
                onChange={(event) => {
                  const value = event.target.value;
                  updateSelectedEdgeData({
                    condition: value || undefined,
                    label: value || undefined,
                  });
                }}
                placeholder="e.g. tool==retrieve or tool==finish"
              />
            </label>
            <p className="muted">
              Supported syntax: <code>key==value</code> or <code>key!=value</code>. For ToolSelector, key is{" "}
              <code>tool</code>.
            </p>
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
    label === "PlannerNode" ||
    label === "ToolSelectorNode" ||
    label === "ToolExecutorNode" ||
    label === "ReflectionNode" ||
    label === "FinalAnswerNode" ||
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
    condition: (edge as any).condition ? String((edge as any).condition) : undefined,
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
  if (type === "PlannerNode") {
    return {
      model: String(data.model ?? "qwen3:8b"),
    };
  }
  if (type === "ToolSelectorNode") {
    return {
      strategy: String(data.strategy ?? "plan_first"),
    };
  }
  if (type === "ToolExecutorNode") {
    const datasetRaw = String(data.dataset_id ?? "").trim();
    const datasetId = datasetRaw ? Number(datasetRaw) : null;
    return {
      final_k: Number(data.final_k ?? data.k ?? 5),
      top_k_bm25: Number(data.top_k_bm25 ?? 8),
      top_k_dense: Number(data.top_k_dense ?? 8),
      rerank_enabled: Boolean(data.rerank_enabled ?? true),
      model: String(data.model ?? "qwen3:8b"),
      ...(datasetId !== null && Number.isFinite(datasetId) ? { dataset_id: datasetId } : {}),
    };
  }
  if (type === "ReflectionNode") {
    return {
      max_loops: Number(data.max_loops ?? 3),
      model: String(data.model ?? "qwen3:8b"),
    };
  }
  if (type === "FinalAnswerNode") {
    return {
      model: String(data.model ?? "qwen3:8b"),
    };
  }
  return {};
}

function eventLabel(event: WorkflowRunStreamEvent): string {
  if (event.type === "started") {
    return `Started (${event.start_nodes.join(", ")})`;
  }
  if (event.type === "node_start") {
    return `Step ${event.step} - Running ${event.node_type} (${event.node_id})`;
  }
  if (event.type === "node_complete") {
    const latency = typeof event.latency_ms === "number" ? ` in ${event.latency_ms}ms` : "";
    return `Step ${event.step} - Completed ${event.node_type} (${event.node_id})${latency}`;
  }
  if (event.type === "edge_traversed") {
    return `Route ${event.source} -> ${event.target}${event.condition ? ` [${event.condition}]` : ""}`;
  }
  if (event.type === "done") {
    return "Workflow done";
  }
  return `Error: ${event.detail}`;
}

function JsonTree({ data }: { data: unknown }) {
  if (data === null || data === undefined) {
    return <span className="json-scalar">null</span>;
  }
  if (typeof data === "string" || typeof data === "number" || typeof data === "boolean") {
    return <span className="json-scalar">{String(data)}</span>;
  }
  if (Array.isArray(data)) {
    if (data.length === 0) return <span className="json-scalar">[]</span>;
    return (
      <ul className="json-tree">
        {data.map((item, idx) => (
          <li key={idx}>
            <details>
              <summary>[{idx}]</summary>
              <JsonTree data={item} />
            </details>
          </li>
        ))}
      </ul>
    );
  }
  const entries = Object.entries(data as Record<string, unknown>);
  if (entries.length === 0) return <span className="json-scalar">{`{}`}</span>;
  return (
    <ul className="json-tree">
      {entries.map(([key, value]) => {
        const isComplex =
          value !== null &&
          (Array.isArray(value) || typeof value === "object");
        if (!isComplex) {
          return (
            <li key={key}>
              <strong>{key}:</strong> <JsonTree data={value} />
            </li>
          );
        }
        return (
          <li key={key}>
            <details>
              <summary>{key}</summary>
              <JsonTree data={value} />
            </details>
          </li>
        );
      })}
    </ul>
  );
}
