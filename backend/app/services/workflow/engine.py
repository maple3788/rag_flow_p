from collections import defaultdict, deque

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.schemas import (
    SourceChunk,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRunRequest,
    WorkflowRunResponse,
)
from app.services.workflow.nodes import WorkflowContext, build_node_executor


def run_workflow(payload: WorkflowRunRequest, db: Session) -> WorkflowRunResponse:
    if not payload.nodes:
        raise HTTPException(status_code=400, detail="Workflow must include nodes")

    node_map = {node.id: node for node in payload.nodes}
    if len(node_map) != len(payload.nodes):
        raise HTTPException(status_code=400, detail="Duplicate node IDs are not allowed")

    _validate_edges(payload.edges, node_map)
    order = _topological_sort(payload.nodes, payload.edges)

    parents_by_node = _group_parents(payload.edges)
    outputs: dict[str, dict] = {}

    for node_id in order:
        node = node_map[node_id]
        parent_ids = parents_by_node.get(node_id, [])
        input_text = "\n".join(
            outputs[parent_id].get("text", "")
            for parent_id in parent_ids
            if parent_id in outputs
        ).strip()

        context = WorkflowContext(
            db=db,
            incoming_text=input_text,
            last_query=_resolve_last_query(parent_ids, outputs),
            last_sources=_resolve_last_sources(parent_ids, outputs),
        )
        executor = build_node_executor(node)
        outputs[node_id] = executor.execute(context)

    output_node_ids = [node.id for node in payload.nodes if node.type == "OutputNode"]
    if not output_node_ids:
        raise HTTPException(status_code=400, detail="Workflow must include an OutputNode")

    last_output_id = output_node_ids[-1]
    final_output = outputs.get(last_output_id, {}).get("text", "")
    return WorkflowRunResponse(output=final_output, node_outputs=outputs)


def _validate_edges(edges: list[WorkflowEdge], node_map: dict[str, WorkflowNode]) -> None:
    for edge in edges:
        if edge.source not in node_map or edge.target not in node_map:
            raise HTTPException(
                status_code=400,
                detail=f"Edge '{edge.id}' references unknown source/target node",
            )


def _group_parents(edges: list[WorkflowEdge]) -> dict[str, list[str]]:
    parents: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        parents[edge.target].append(edge.source)
    return parents


def _topological_sort(nodes: list[WorkflowNode], edges: list[WorkflowEdge]) -> list[str]:
    indegree: dict[str, int] = {node.id: 0 for node in nodes}
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        graph[edge.source].append(edge.target)
        indegree[edge.target] += 1

    queue = deque([node_id for node_id, degree in indegree.items() if degree == 0])
    order: list[str] = []

    while queue:
        current = queue.popleft()
        order.append(current)
        for nxt in graph[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(nodes):
        raise HTTPException(status_code=400, detail="Workflow graph must be a DAG")
    return order


def _resolve_last_query(parent_ids: list[str], outputs: dict[str, dict]) -> str:
    for parent_id in reversed(parent_ids):
        query = str(outputs.get(parent_id, {}).get("query", "")).strip()
        if query:
            return query
    return ""


def _resolve_last_sources(
    parent_ids: list[str], outputs: dict[str, dict]
) -> list[SourceChunk] | None:
    for parent_id in reversed(parent_ids):
        candidate = outputs.get(parent_id, {}).get("sources")
        if candidate:
            return [SourceChunk.model_validate(item) for item in candidate]
    return None
