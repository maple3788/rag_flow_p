from collections import defaultdict, deque
from collections.abc import Generator
from time import perf_counter
from typing import Any

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.schemas import SourceChunk, WorkflowEdge, WorkflowNode, WorkflowRunRequest, WorkflowRunResponse
from app.services.workflow.nodes import WorkflowContext, build_node_executor


def run_workflow(payload: WorkflowRunRequest, db: Session) -> WorkflowRunResponse:
    final_response: WorkflowRunResponse | None = None
    for event in iter_workflow_events(payload=payload, db=db):
        if event.get("type") == "done":
            final_response = WorkflowRunResponse(
                output=str(event.get("output", "")),
                node_outputs=dict(event.get("node_outputs", {})),
                route_trace=list(event.get("route_trace", [])),
            )
    if final_response is None:
        raise HTTPException(status_code=500, detail="Workflow completed without final output")
    return final_response


def iter_workflow_events(payload: WorkflowRunRequest, db: Session) -> Generator[dict[str, Any], None, None]:
    if not payload.nodes:
        raise HTTPException(status_code=400, detail="Workflow must include nodes")

    node_map = {node.id: node for node in payload.nodes}
    if len(node_map) != len(payload.nodes):
        raise HTTPException(status_code=400, detail="Duplicate node IDs are not allowed")

    _validate_edges(payload.edges, node_map)
    parents_by_node = _group_parents(payload.edges)
    edges_by_source = _group_outgoing_edges(payload.edges)
    outputs: dict[str, dict] = {}
    route_trace: list[dict] = []
    shared_state: dict[str, Any] = {
        "query": "",
        "history": [],
        "tool_results": [],
        "plan": {},
        "plan_cursor": 0,
        "reflection_loops": 0,
    }

    start_nodes = [node.id for node in payload.nodes if not parents_by_node.get(node.id)]
    if not start_nodes:
        raise HTTPException(status_code=400, detail="Workflow must include at least one start node")
    yield {"type": "started", "start_nodes": start_nodes}

    queue = deque(start_nodes)
    run_counts: dict[str, int] = defaultdict(int)
    global_steps = 0
    max_global_steps = 200
    max_node_repeats = 50

    while queue:
        node_id = queue.popleft()
        node = node_map[node_id]
        run_counts[node_id] += 1
        global_steps += 1
        if run_counts[node_id] > max_node_repeats:
            raise HTTPException(status_code=400, detail=f"Node '{node_id}' exceeded max repeats")
        if global_steps > max_global_steps:
            raise HTTPException(status_code=400, detail="Workflow exceeded max runtime steps")

        parent_ids = parents_by_node.get(node_id, [])
        started_at = perf_counter()
        yield {
            "type": "node_start",
            "step": global_steps,
            "node_id": node_id,
            "node_type": node.type,
            "parents": parent_ids,
        }
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
            shared_state=shared_state,
        )
        executor = build_node_executor(node)
        result = executor.execute(context)
        outputs[node_id] = result
        latency_ms = round((perf_counter() - started_at) * 1000, 3)
        yield {
            "type": "node_complete",
            "step": global_steps,
            "node_id": node_id,
            "node_type": node.type,
            "latency_ms": latency_ms,
            "preview": str(result.get("text", ""))[:200],
            "details": {
                "input": {
                    "incoming_text": input_text,
                    "last_query": context.last_query,
                    "parent_ids": parent_ids,
                    "retrieved_sources": _make_json_safe(
                        context.last_sources if context.last_sources else _latest_shared_sources(shared_state)
                    ),
                },
                "llm": _make_json_safe(result.get("llm_trace")),
                "variables": {
                    "output": _make_json_safe({k: v for k, v in result.items() if k != "llm_trace"}),
                    "state": _build_state_snapshot(shared_state),
                },
            },
        }

        outgoing = edges_by_source.get(node_id, [])
        if not outgoing:
            continue

        selected_edges = _select_outgoing_edges(
            node=node,
            outgoing_edges=outgoing,
            result=result,
            node_map=node_map,
        )
        for edge in selected_edges:
            queue.append(edge.target)
            yield {
                "type": "edge_traversed",
                "edge_id": edge.id,
                "source": edge.source,
                "target": edge.target,
                "condition": edge.condition,
            }
            route_trace.append(
                {
                    "edge_id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "condition": edge.condition,
                }
            )

    output_node_ids = [node.id for node in payload.nodes if node.type == "OutputNode"]
    if not output_node_ids:
        raise HTTPException(status_code=400, detail="Workflow must include an OutputNode")

    last_output_id = output_node_ids[-1]
    final_output = outputs.get(last_output_id, {}).get("text", "")
    response = WorkflowRunResponse(output=final_output, node_outputs=outputs, route_trace=route_trace)
    yield {
        "type": "done",
        "output": response.output,
        "node_outputs": response.node_outputs,
        "route_trace": response.route_trace,
    }


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


def _group_outgoing_edges(edges: list[WorkflowEdge]) -> dict[str, list[WorkflowEdge]]:
    grouped: dict[str, list[WorkflowEdge]] = defaultdict(list)
    for edge in edges:
        grouped[edge.source].append(edge)
    return grouped


def _select_outgoing_edges(
    node: WorkflowNode,
    outgoing_edges: list[WorkflowEdge],
    result: dict,
    node_map: dict[str, WorkflowNode],
) -> list[WorkflowEdge]:
    if node.type == "ReflectionNode":
        should_continue = bool(result.get("continue", False))
        selected: list[WorkflowEdge] = []
        for edge in outgoing_edges:
            target_type = node_map.get(edge.target).type if edge.target in node_map else ""
            if should_continue and target_type == "PlannerNode":
                selected.append(edge)
            elif (not should_continue) and target_type != "PlannerNode":
                selected.append(edge)
        return selected

    if node.type == "ToolSelectorNode":
        tool = str(result.get("selection", {}).get("tool", "")).strip().lower()
        conditional = [edge for edge in outgoing_edges if edge.condition]
        if conditional:
            matched = [
                edge
                for edge in conditional
                if _evaluate_edge_condition(edge.condition or "", {"tool": tool})
            ]
            if matched:
                return matched
        return [edge for edge in outgoing_edges if not edge.condition] or outgoing_edges

    return outgoing_edges


def _evaluate_edge_condition(condition: str, values: dict[str, Any]) -> bool:
    expr = condition.strip().lower()
    if "==" in expr:
        key, value = [part.strip() for part in expr.split("==", 1)]
        return str(values.get(key, "")).strip().lower() == value
    if "!=" in expr:
        key, value = [part.strip() for part in expr.split("!=", 1)]
        return str(values.get(key, "")).strip().lower() != value
    return False


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


def _build_state_snapshot(shared_state: dict[str, Any]) -> dict[str, Any]:
    selected_tool = shared_state.get("selected_tool")
    plan = shared_state.get("plan")
    tool_results = shared_state.get("tool_results") or []
    history = shared_state.get("history") or []
    return {
        "query": str(shared_state.get("query", "")),
        "plan_cursor": int(shared_state.get("plan_cursor", 0) or 0),
        "reflection_loops": int(shared_state.get("reflection_loops", 0) or 0),
        "selected_tool": _make_json_safe(selected_tool) if selected_tool else None,
        "plan": _make_json_safe(plan) if plan else None,
        "tool_results_count": len(tool_results),
        "history_count": len(history),
        "latest_tool_result": _make_json_safe(tool_results[-1]) if tool_results else None,
        "latest_sources": _make_json_safe(_latest_shared_sources(shared_state)),
    }


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _make_json_safe(val) for key, val in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return _make_json_safe(value.model_dump())
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def _latest_shared_sources(shared_state: dict[str, Any]) -> list[Any]:
    latest_sources = shared_state.get("latest_sources")
    if isinstance(latest_sources, list) and latest_sources:
        return latest_sources
    tool_results = shared_state.get("tool_results", [])
    if isinstance(tool_results, list):
        for item in reversed(tool_results):
            if str(item.get("tool", "")).strip().lower() != "retrieve":
                continue
            sources = item.get("sources")
            if isinstance(sources, list) and sources:
                return sources
    return []
