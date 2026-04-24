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

type AgentSubgraphData = {
  agent_data: Record<string, unknown>;
  internal_nodes: WorkflowNode[];
  internal_edges: WorkflowEdge[];
  incoming_edges: WorkflowEdge[];
  outgoing_edges: WorkflowEdge[];
};

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

type WorkflowTemplate = {
  id: string;
  title: string;
  mainIdea: string;
  nodes: Node[];
  edges: Edge[];
};

const workflowTemplates: WorkflowTemplate[] = [
  {
    id: "agentic-loop",
    title: "Agentic Loop",
    mainIdea: "Plan -> tool -> reflect until sufficient, then answer.",
    nodes: defaultNodes,
    edges: defaultEdges,
  },
  {
    id: "retrieve-then-answer",
    title: "Direct RAG",
    mainIdea: "Single retrieval pass then final answer.",
    nodes: [
      { id: "input-a", type: "default", position: { x: 80, y: 120 }, data: { label: "InputNode", query: "Ask your question here" } },
      { id: "executor-a", type: "default", position: { x: 360, y: 120 }, data: { label: "ToolExecutorNode", final_k: 6, top_k_bm25: 12, top_k_dense: 12, rerank_enabled: true, model: "qwen3:8b", dataset_id: "" } },
      { id: "final-a", type: "default", position: { x: 660, y: 120 }, data: { label: "FinalAnswerNode", model: "qwen3:8b" } },
      { id: "output-a", type: "default", position: { x: 920, y: 120 }, data: { label: "OutputNode" } },
    ],
    edges: [
      { id: "e-a-1", source: "input-a", target: "executor-a" },
      { id: "e-a-2", source: "executor-a", target: "final-a" },
      { id: "e-a-3", source: "final-a", target: "output-a" },
    ],
  },
  {
    id: "plan-before-retrieve",
    title: "Plan-Guided RAG",
    mainIdea: "Create a plan first, then retrieve and answer.",
    nodes: [
      { id: "input-b", type: "default", position: { x: 80, y: 140 }, data: { label: "InputNode", query: "Ask your question here" } },
      { id: "planner-b", type: "default", position: { x: 320, y: 140 }, data: { label: "PlannerNode", model: "qwen3:8b" } },
      { id: "selector-b", type: "default", position: { x: 560, y: 140 }, data: { label: "ToolSelectorNode", strategy: "plan_first" } },
      { id: "executor-b", type: "default", position: { x: 800, y: 140 }, data: { label: "ToolExecutorNode", final_k: 5, top_k_bm25: 10, top_k_dense: 10, rerank_enabled: true, model: "qwen3:8b", dataset_id: "" } },
      { id: "final-b", type: "default", position: { x: 1040, y: 140 }, data: { label: "FinalAnswerNode", model: "qwen3:8b" } },
      { id: "output-b", type: "default", position: { x: 1260, y: 140 }, data: { label: "OutputNode" } },
    ],
    edges: [
      { id: "e-b-1", source: "input-b", target: "planner-b" },
      { id: "e-b-2", source: "planner-b", target: "selector-b" },
      { id: "e-b-3", source: "selector-b", target: "executor-b", label: "tool!=finish", ...( { condition: "tool!=finish" } as any ) },
      { id: "e-b-4", source: "executor-b", target: "final-b" },
      { id: "e-b-5", source: "final-b", target: "output-b" },
    ],
  },
  {
    id: "self-correcting-rag",
    title: "Self-Correcting RAG",
    mainIdea: "Retrieve, reflect on quality, retry if needed.",
    nodes: [
      { id: "input-c", type: "default", position: { x: 80, y: 120 }, data: { label: "InputNode", query: "Ask your question here" } },
      { id: "planner-c", type: "default", position: { x: 320, y: 120 }, data: { label: "PlannerNode", model: "qwen3:8b" } },
      { id: "selector-c", type: "default", position: { x: 560, y: 120 }, data: { label: "ToolSelectorNode", strategy: "plan_first" } },
      { id: "executor-c", type: "default", position: { x: 800, y: 120 }, data: { label: "ToolExecutorNode", final_k: 8, top_k_bm25: 16, top_k_dense: 16, rerank_enabled: true, model: "qwen3:8b", dataset_id: "" } },
      { id: "reflection-c", type: "default", position: { x: 560, y: 280 }, data: { label: "ReflectionNode", max_loops: 3, model: "qwen3:8b" } },
      { id: "final-c", type: "default", position: { x: 800, y: 280 }, data: { label: "FinalAnswerNode", model: "qwen3:8b" } },
      { id: "output-c", type: "default", position: { x: 1040, y: 280 }, data: { label: "OutputNode" } },
    ],
    edges: [
      { id: "e-c-1", source: "input-c", target: "planner-c" },
      { id: "e-c-2", source: "planner-c", target: "selector-c" },
      { id: "e-c-3", source: "selector-c", target: "executor-c", label: "tool!=finish", ...( { condition: "tool!=finish" } as any ) },
      { id: "e-c-4", source: "executor-c", target: "reflection-c" },
      { id: "e-c-5", source: "reflection-c", target: "planner-c", label: "continue=true" },
      { id: "e-c-6", source: "reflection-c", target: "final-c", label: "continue=false" },
      { id: "e-c-7", source: "final-c", target: "output-c" },
    ],
  },
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
  const [agentSubgraphs, setAgentSubgraphs] = useState<Record<string, AgentSubgraphData>>({});

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

  function loadTemplate(template: WorkflowTemplate) {
    setNodes(template.nodes.map((node) => ({ ...node, selected: false })));
    setEdges(template.edges.map((edge) => ({ ...edge, selected: false, animated: false })));
    setActiveEdgeIds([]);
    setCurrentNodeId(null);
    setStreamEvents([]);
    setResult("");
    setError("");
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
      const nextAgentSubgraphs: Record<string, AgentSubgraphData> = {};
      for (const node of nextNodes) {
        if (node.data?.label !== "AgentNode") continue;
        const subgraph = (node.data as Record<string, unknown>)?.agent_subgraph;
        if (subgraph && typeof subgraph === "object") {
          nextAgentSubgraphs[node.id] = subgraph as AgentSubgraphData;
        }
      }
      setAgentSubgraphs(nextAgentSubgraphs);
      setActiveEdgeIds([]);
      setError("");
    } catch {
      setError("Invalid workflow JSON");
    }
  }

  function expandAgentNode(agentId: string) {
    const agentNode = nodes.find((node) => node.id === agentId);
    if (!agentNode || nodeTypeFromLabel(agentNode.data?.label) !== "AgentNode") return;
    const metadata =
      agentSubgraphs[agentId] ?? buildDefaultAgentSubgraph(agentNode as Node);
    const connectedIncoming: WorkflowEdge[] = edges
      .filter((edge) => edge.target === agentId && edge.source !== agentId)
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: agentId,
        condition: (edge as any).condition ? String((edge as any).condition) : undefined,
      }));
    const connectedOutgoing: WorkflowEdge[] = edges
      .filter((edge) => edge.source === agentId && edge.target !== agentId)
      .map((edge) => ({
        id: edge.id,
        source: agentId,
        target: edge.target,
        condition: (edge as any).condition ? String((edge as any).condition) : undefined,
      }));
    const effectiveMetadata: AgentSubgraphData = {
      ...metadata,
      incoming_edges: connectedIncoming.length ? connectedIncoming : metadata.incoming_edges,
      outgoing_edges: connectedOutgoing.length ? connectedOutgoing : metadata.outgoing_edges,
    };
    const entryId = metadata.internal_nodes[0]?.id;
    const exitId = metadata.internal_nodes[metadata.internal_nodes.length - 1]?.id;
    if (!entryId || !exitId) return;

    const internalNodes: Node[] = effectiveMetadata.internal_nodes.map((node) => ({
      id: node.id,
      type: "default",
      position: node.position ?? { x: 0, y: 0 },
      data: {
        ...node.data,
        label: node.type,
        parent_agent_id: agentId,
      },
    }));
    const internalEdges: Edge[] = effectiveMetadata.internal_edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      label: edge.condition ?? undefined,
      ...(edge.condition ? ({ condition: edge.condition } as any) : {}),
    }));
    const incomingEdges: Edge[] = effectiveMetadata.incoming_edges.map((edge) => ({
      id: `${edge.id}__exp_${Date.now()}`,
      source: edge.source,
      target: entryId,
      label: edge.condition ?? undefined,
      ...(edge.condition ? ({ condition: edge.condition } as any) : {}),
    }));
    const outgoingEdges: Edge[] = effectiveMetadata.outgoing_edges.map((edge) => ({
      id: `${edge.id}__exp_${Date.now()}_out`,
      source: exitId,
      target: edge.target,
      label: edge.condition ?? undefined,
      ...(edge.condition ? ({ condition: edge.condition } as any) : {}),
    }));

    setNodes((prev) => [...prev.filter((node) => node.id !== agentId), ...internalNodes]);
    setEdges((prev) => [
      ...prev.filter((edge) => edge.source !== agentId && edge.target !== agentId),
      ...internalEdges,
      ...incomingEdges,
      ...outgoingEdges,
    ]);
    setAgentSubgraphs((prev) => ({ ...prev, [agentId]: effectiveMetadata }));
  }

  function collapseAgentNode(agentId: string) {
    const metadata = agentSubgraphs[agentId];
    if (!metadata) return;
    const internalIds = new Set(metadata.internal_nodes.map((node) => node.id));
    const entryId = metadata.internal_nodes[0]?.id;
    const exitId = metadata.internal_nodes[metadata.internal_nodes.length - 1]?.id;
    if (!entryId || !exitId) return;

    const liveInternalNodes = nodes.filter((node) => internalIds.has(node.id));
    const nextInternalNodes: WorkflowNode[] = liveInternalNodes.map((node) => ({
      id: node.id,
      type: nodeTypeFromLabel(node.data?.label) ?? "PlannerNode",
      data: normalizeNodeData(node.data ?? {}, nodeTypeFromLabel(node.data?.label) ?? "PlannerNode"),
      position: node.position,
    }));
    const liveEdges = edges;
    const nextInternalEdges: WorkflowEdge[] = liveEdges
      .filter((edge) => internalIds.has(edge.source) && internalIds.has(edge.target))
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        condition: (edge as any).condition ? String((edge as any).condition) : undefined,
      }));
    const nextIncoming: WorkflowEdge[] = liveEdges
      .filter((edge) => edge.target === entryId && !internalIds.has(edge.source))
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: agentId,
        condition: (edge as any).condition ? String((edge as any).condition) : undefined,
      }));
    const nextOutgoing: WorkflowEdge[] = liveEdges
      .filter((edge) => edge.source === exitId && !internalIds.has(edge.target))
      .map((edge) => ({
        id: edge.id,
        source: agentId,
        target: edge.target,
        condition: (edge as any).condition ? String((edge as any).condition) : undefined,
      }));

    const plannerNode = liveInternalNodes.find((node) => node.id === entryId);
    const restoredAgentData: Record<string, unknown> = {
      ...metadata.agent_data,
      label: "AgentNode",
      agent_subgraph: {
        ...metadata,
        internal_nodes: nextInternalNodes.length ? nextInternalNodes : metadata.internal_nodes,
        internal_edges: nextInternalEdges.length ? nextInternalEdges : metadata.internal_edges,
        incoming_edges: nextIncoming.length ? nextIncoming : metadata.incoming_edges,
        outgoing_edges: nextOutgoing.length ? nextOutgoing : metadata.outgoing_edges,
      },
    };
    const restoredAgentNode: Node = {
      id: agentId,
      type: "default",
      position: plannerNode?.position ?? metadata.internal_nodes[0]?.position ?? { x: 120, y: 120 },
      data: restoredAgentData,
    };

    setNodes((prev) => [...prev.filter((node) => !internalIds.has(node.id)), restoredAgentNode]);
    setEdges((prev) => [
      ...prev.filter((edge) => !internalIds.has(edge.source) && !internalIds.has(edge.target)),
      ...nextIncoming.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.condition ?? undefined,
        ...(edge.condition ? ({ condition: edge.condition } as any) : {}),
      })),
      ...nextOutgoing.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        label: edge.condition ?? undefined,
        ...(edge.condition ? ({ condition: edge.condition } as any) : {}),
      })),
    ]);
    setAgentSubgraphs((prev) => ({
      ...prev,
      [agentId]: restoredAgentData.agent_subgraph as AgentSubgraphData,
    }));
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
      <div className="workflow-toolbar">
        {workflowTemplates.map((template) => (
          <button
            key={template.id}
            className="button secondary"
            onClick={() => loadTemplate(template)}
            title={template.mainIdea}
          >
            {template.title}
          </button>
        ))}
      </div>
      <p className="muted">
        Template ideas:{" "}
        {workflowTemplates.map((template) => `${template.title} - ${template.mainIdea}`).join(" | ")}
      </p>

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
                <button
                  className="button secondary"
                  onClick={() => expandAgentNode(selectedNode.id)}
                >
                  Expand Agent
                </button>
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
            {selectedNode.data?.parent_agent_id && (
              <button
                className="button secondary"
                onClick={() => collapseAgentNode(String(selectedNode.data?.parent_agent_id))}
              >
                Collapse Agent
              </button>
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
    const normalized: Record<string, unknown> = {
      query: String(data.query ?? ""),
      k: Number(data.k ?? 5),
      max_steps: Number(data.max_steps ?? 5),
      use_web_search: Boolean(data.use_web_search ?? false),
      model: String(data.model ?? "qwen3:8b"),
    };
    if (data.agent_subgraph && typeof data.agent_subgraph === "object") {
      normalized.agent_subgraph = data.agent_subgraph;
    }
    return normalized;
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

function buildDefaultAgentSubgraph(agentNode: Node): AgentSubgraphData {
  const agentId = agentNode.id;
  const x = agentNode.position.x;
  const y = agentNode.position.y;
  const internalNodes: WorkflowNode[] = [
    {
      id: `${agentId}__planner`,
      type: "PlannerNode",
      position: { x: x - 120, y: y - 60 },
      data: { model: "qwen3:8b" },
    },
    {
      id: `${agentId}__selector`,
      type: "ToolSelectorNode",
      position: { x: x + 120, y: y - 60 },
      data: { strategy: "plan_first" },
    },
    {
      id: `${agentId}__executor`,
      type: "ToolExecutorNode",
      position: { x: x + 360, y: y - 60 },
      data: { final_k: 5, top_k_bm25: 8, top_k_dense: 8, rerank_enabled: true, model: "qwen3:8b" },
    },
    {
      id: `${agentId}__reflection`,
      type: "ReflectionNode",
      position: { x: x + 120, y: y + 120 },
      data: { max_loops: 3, model: "qwen3:8b" },
    },
    {
      id: `${agentId}__final`,
      type: "FinalAnswerNode",
      position: { x: x + 360, y: y + 120 },
      data: { model: "qwen3:8b" },
    },
  ];
  const internalEdges: WorkflowEdge[] = [
    { id: `${agentId}__e1`, source: `${agentId}__planner`, target: `${agentId}__selector` },
    {
      id: `${agentId}__e2`,
      source: `${agentId}__selector`,
      target: `${agentId}__executor`,
      condition: "tool!=finish",
    },
    { id: `${agentId}__e3`, source: `${agentId}__executor`, target: `${agentId}__reflection` },
    { id: `${agentId}__e4`, source: `${agentId}__reflection`, target: `${agentId}__planner` },
    { id: `${agentId}__e5`, source: `${agentId}__reflection`, target: `${agentId}__final` },
  ];
  return {
    agent_data: normalizeNodeData(agentNode.data as Record<string, unknown>, "AgentNode"),
    internal_nodes: internalNodes,
    internal_edges: internalEdges,
    incoming_edges: [],
    outgoing_edges: [],
  };
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
