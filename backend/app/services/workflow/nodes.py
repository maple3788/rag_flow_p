from dataclasses import dataclass, field
import json
from typing import Any, TypedDict

import requests
from fastapi import HTTPException
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.config import settings
from app.schemas import SourceChunk, WorkflowNode
from app.services.chat import generate_answer
from app.services.retrieval import retrieve_similar_chunks


@dataclass
class WorkflowContext:
    db: Session
    incoming_text: str = ""
    last_query: str = ""
    last_sources: list[SourceChunk] | None = None
    shared_state: dict[str, Any] = field(default_factory=dict)


class BaseWorkflowNode:
    def __init__(self, node: WorkflowNode):
        self.node = node

    def execute(self, context: WorkflowContext) -> dict:
        raise NotImplementedError


class InputNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        query = str(self.node.data.get("query", "")).strip()
        if not query:
            query = context.incoming_text.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail=f"InputNode '{self.node.id}' requires a query",
            )
        context.last_query = query
        return {"text": query}


class RetrieverNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        query = context.incoming_text.strip() or context.last_query.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail=f"RetrieverNode '{self.node.id}' did not receive query input",
            )
        k = int(self.node.data.get("k", 5))
        sources = retrieve_similar_chunks(db=context.db, query=query, k=k)
        context.last_query = query
        context.last_sources = sources
        context.shared_state["latest_sources"] = [source.model_dump() for source in sources]
        joined_context = "\n\n".join(src.content for src in sources)
        return {
            "text": joined_context,
            "query": query,
            "sources": [source.model_dump() for source in sources],
        }


class LLMNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        query = context.last_query.strip() or context.incoming_text.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail=f"LLMNode '{self.node.id}' did not receive query input",
            )

        template = str(
            self.node.data.get(
                "template",
                "Use the retrieved context to answer the user question.",
            )
        )
        sources = context.last_sources or []

        enriched_query = f"{template}\n\nQuestion: {query}"
        model = self.node.data.get("model")
        model_name = str(model) if model else None
        answer = generate_answer(
            query=enriched_query,
            sources=sources,
            model=model_name,
        )
        return {
            "text": answer,
            "sources": [source.model_dump() for source in sources],
            "llm_trace": {
                "provider": "generate_answer",
                "model": model_name or settings.chat_model,
                "input": {
                    "query": enriched_query,
                    "source_count": len(sources),
                    "source_preview": [source.content[:280] for source in sources[:3]],
                },
                "output": answer,
            },
        }


class OutputNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        if not context.incoming_text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"OutputNode '{self.node.id}' has no input",
            )
        return {"text": context.incoming_text}


class AgentState(TypedDict):
    query: str
    model: str
    max_steps: int
    use_web_search: bool
    step: int
    action: str
    action_input: str
    tool_results: list[dict[str, Any]]
    retrieved_sources: list[dict[str, Any]]
    final_answer: str
    trace: list[str]


class AgentNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        query = str(self.node.data.get("query", "")).strip() or context.incoming_text.strip()
        if not query:
            raise HTTPException(
                status_code=400,
                detail=f"AgentNode '{self.node.id}' requires query input",
            )

        model = str(self.node.data.get("model", settings.chat_model))
        max_steps = int(self.node.data.get("max_steps", 5))
        use_web_search = bool(self.node.data.get("use_web_search", False))

        graph = self._build_agent_graph(context)
        compiled = graph.compile()
        state: AgentState = {
            "query": query,
            "model": model,
            "max_steps": max_steps,
            "use_web_search": use_web_search,
            "step": 0,
            "action": "",
            "action_input": "",
            "tool_results": [],
            "retrieved_sources": [],
            "final_answer": "",
            "trace": [],
        }

        result = compiled.invoke(state)
        final_answer = str(result.get("final_answer", "")).strip()
        if not final_answer:
            raise HTTPException(
                status_code=500,
                detail=f"AgentNode '{self.node.id}' failed to produce final answer",
            )

        sources = [
            SourceChunk.model_validate(item)
            for item in result.get("retrieved_sources", [])
        ]
        if sources:
            context.last_sources = sources
        context.last_query = query
        return {
            "text": final_answer,
            "query": query,
            "sources": [source.model_dump() for source in sources],
            "agent_trace": result.get("trace", []),
            "agent_tools": result.get("tool_results", []),
        }

    def _build_agent_graph(self, context: WorkflowContext) -> StateGraph:
        graph = StateGraph(AgentState)

        def decide(state: AgentState) -> AgentState:
            if state["step"] >= state["max_steps"]:
                state["action"] = "finish"
                state["trace"].append("Max steps reached; finalizing.")
                return state

            decision = self._decide_next_action(state)
            state["action"] = decision.get("action", "finish")
            state["action_input"] = decision.get("input", "")
            if decision.get("final_answer"):
                state["final_answer"] = decision["final_answer"]
            state["trace"].append(
                f"Step {state['step'] + 1}: decided {state['action']} ({state['action_input']})"
            )
            return state

        def use_tool(state: AgentState) -> AgentState:
            action = state["action"]
            action_input = state["action_input"] or state["query"]

            if action == "retrieve":
                k = int(self.node.data.get("k", 5))
                sources = retrieve_similar_chunks(db=context.db, query=action_input, k=k)
                state["retrieved_sources"] = [source.model_dump() for source in sources]
                state["tool_results"].append(
                    {
                        "tool": "retriever",
                        "input": action_input,
                        "output": [source.content for source in sources],
                    }
                )
            elif action == "calculate":
                state["tool_results"].append(
                    {
                        "tool": "calculator",
                        "input": action_input,
                        "output": str(_safe_calculate(action_input)),
                    }
                )
            elif action == "web_search":
                if not state["use_web_search"]:
                    state["tool_results"].append(
                        {
                            "tool": "web_search",
                            "input": action_input,
                            "output": "Web search disabled in node settings.",
                        }
                    )
                else:
                    state["tool_results"].append(
                        {
                            "tool": "web_search",
                            "input": action_input,
                            "output": _web_search(action_input),
                        }
                    )
            state["step"] += 1
            return state

        def finalize(state: AgentState) -> AgentState:
            if state["final_answer"]:
                return state
            state["final_answer"] = self._finalize_answer(state)
            return state

        def route_after_decide(state: AgentState) -> str:
            if state["action"] in {"retrieve", "calculate", "web_search"}:
                return "use_tool"
            return "finalize"

        graph.add_node("decide", decide)
        graph.add_node("use_tool", use_tool)
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide",
            route_after_decide,
            {
                "use_tool": "use_tool",
                "finalize": "finalize",
            },
        )
        graph.add_edge("use_tool", "decide")
        graph.add_edge("finalize", END)
        return graph

    def _decide_next_action(self, state: AgentState) -> dict[str, str]:
        tool_state = json.dumps(state["tool_results"][-5:], ensure_ascii=True)
        prompt = (
            "You are an autonomous RAG agent planner.\n"
            "Choose the best next action based on the user query and prior tool results.\n"
            "Allowed actions: retrieve, calculate, web_search, finish.\n"
            "Return JSON only: "
            '{"action":"retrieve|calculate|web_search|finish","input":"...",'
            '"final_answer":"optional"}\n'
            f"User query: {state['query']}\n"
            f"Previous tool results: {tool_state}\n"
            f"Web search enabled: {state['use_web_search']}"
        )
        raw = _ollama_chat(
            model=state["model"],
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        parsed = _parse_json(raw)
        action = str(parsed.get("action", "finish"))
        if action not in {"retrieve", "calculate", "web_search", "finish"}:
            action = "finish"
        return {
            "action": action,
            "input": str(parsed.get("input", "")),
            "final_answer": str(parsed.get("final_answer", "")),
        }

    def _finalize_answer(self, state: AgentState) -> str:
        sources = [
            SourceChunk.model_validate(item)
            for item in state.get("retrieved_sources", [])
        ]
        if sources:
            return generate_answer(
                query=state["query"],
                sources=sources,
                model=state["model"],
            )
        memory_text = "\n".join(
            f"{entry.get('tool')}: {entry.get('output')}" for entry in state["tool_results"]
        )
        prompt = (
            "Produce a concise final answer to the user query using tool outputs.\n"
            f"User query: {state['query']}\n"
            f"Tool outputs:\n{memory_text or '(none)'}"
        )
        return _ollama_chat(
            model=state["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )


class PlannerNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        query = context.incoming_text.strip() or context.last_query.strip()
        if not query:
            query = str(context.shared_state.get("query", "")).strip()
        if not query:
            raise HTTPException(status_code=400, detail=f"PlannerNode '{self.node.id}' requires query input")

        model = str(self.node.data.get("model", settings.chat_model))
        history = context.shared_state.get("history", [])
        history_json = json.dumps(history[-10:], ensure_ascii=True)
        prompt = (
            "You are a planner for an agentic RAG workflow.\n"
            "Return strict JSON with a compact executable plan.\n"
            'Format: {"steps":[{"tool":"retrieve|calculate|api|finish","input":"...","reason":"..."}],'
            '"goal":"...","constraints":["..."]}\n'
            f"Query: {query}\n"
            f"History: {history_json}"
        )
        raw = _ollama_chat(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        plan = _parse_json(raw)
        if not isinstance(plan.get("steps"), list):
            plan = {"steps": [{"tool": "retrieve", "input": query, "reason": "default fallback"}], "goal": query}

        context.shared_state["query"] = query
        context.shared_state["plan"] = plan
        context.shared_state.setdefault("history", []).append({"node": "planner", "plan": plan})
        context.last_query = query
        return {
            "text": json.dumps(plan, ensure_ascii=True),
            "query": query,
            "plan": plan,
            "llm_trace": {
                "provider": "ollama_chat",
                "model": model,
                "input": prompt,
                "output": raw,
            },
        }


class ToolSelectorNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        plan = context.shared_state.get("plan", {})
        steps = plan.get("steps", []) if isinstance(plan, dict) else []
        tool_results = context.shared_state.get("tool_results", [])
        cursor = int(context.shared_state.get("plan_cursor", 0))

        selected = None
        while cursor < len(steps):
            step = steps[cursor]
            cursor += 1
            if isinstance(step, dict):
                selected = step
                break
        context.shared_state["plan_cursor"] = cursor

        if not selected:
            fallback_query = str(context.shared_state.get("query", context.last_query)).strip()
            selected = {"tool": "retrieve", "input": fallback_query, "reason": "fallback selector"}

        tool = str(selected.get("tool", "retrieve")).strip().lower()
        if tool not in {"retrieve", "calculate", "api", "finish"}:
            tool = "retrieve"
        selection = {
            "tool": tool,
            "args": str(selected.get("input", context.shared_state.get("query", ""))).strip(),
            "reason": str(selected.get("reason", "")),
            "seen_results": len(tool_results),
        }
        context.shared_state["selected_tool"] = selection
        context.shared_state.setdefault("history", []).append({"node": "tool_selector", "selection": selection})
        return {"text": json.dumps(selection, ensure_ascii=True), "selection": selection}


class ToolExecutorNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        selection = context.shared_state.get("selected_tool", {})
        tool = str(selection.get("tool", "retrieve")).strip().lower()
        args = str(selection.get("args", context.last_query)).strip()
        model = str(self.node.data.get("model", settings.chat_model))
        raw_dataset_id = self.node.data.get("dataset_id")
        dataset_id: int | None = None
        if raw_dataset_id not in (None, "", "null"):
            try:
                dataset_id = int(raw_dataset_id)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail=f"ToolExecutorNode '{self.node.id}' has invalid dataset_id",
                )

        result: dict[str, Any]
        if tool == "retrieve":
            final_k = int(self.node.data.get("final_k", self.node.data.get("k", 5)))
            top_k_bm25 = _optional_int(self.node.data.get("top_k_bm25"))
            top_k_dense = _optional_int(self.node.data.get("top_k_dense"))
            rerank_enabled = bool(self.node.data.get("rerank_enabled", True))
            sources = retrieve_similar_chunks(
                db=context.db,
                query=args or context.last_query,
                k=final_k,
                dataset_id=dataset_id,
                top_k_bm25=top_k_bm25,
                top_k_dense=top_k_dense,
                rerank_enabled=rerank_enabled,
            )
            context.last_sources = sources
            context.shared_state["latest_sources"] = [source.model_dump() for source in sources]
            result = {
                "tool": "retrieve",
                "input": args,
                "dataset_id": dataset_id,
                "final_k": final_k,
                "top_k_bm25": top_k_bm25,
                "top_k_dense": top_k_dense,
                "rerank_enabled": rerank_enabled,
                "output": [source.content for source in sources],
                "sources": [source.model_dump() for source in sources],
            }
        elif tool == "calculate":
            calc = _safe_calculate(args)
            result = {"tool": "calculate", "input": args, "output": str(calc)}
        elif tool == "api":
            result = {"tool": "api", "input": args, "output": _web_search(args)}
        else:
            result = {"tool": "finish", "input": args, "output": "No tool execution needed."}

        context.shared_state.setdefault("tool_results", []).append(result)
        context.shared_state.setdefault("history", []).append({"node": "tool_executor", "result": result})
        return {"text": str(result.get("output", "")), "tool_result": result}


class ReflectionNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        query = str(context.shared_state.get("query", context.last_query)).strip()
        tool_results = context.shared_state.get("tool_results", [])
        retrieval_quality = _assess_retrieval_quality(query=query, tool_results=tool_results)
        relevance_judgment = _judge_retrieval_relevance(
            query=query,
            tool_results=tool_results,
            model="llama3.2:latest",
        )
        max_loops = int(self.node.data.get("max_loops", 3))
        loop_count = int(context.shared_state.get("reflection_loops", 0))

        if loop_count >= max_loops:
            decision = {"continue": False, "reason": "max_loops reached", "refined_query": query}
            llm_trace: dict[str, Any] | None = None
        elif retrieval_quality["has_retrieve_call"] and retrieval_quality["non_empty_docs"] == 0:
            decision = {
                "continue": True,
                "reason": "retrieval returned no usable docs",
                "refined_query": query,
            }
            llm_trace = None
        elif relevance_judgment["available"] and not relevance_judgment["relevant"]:
            decision = {
                "continue": True,
                "reason": f"llama3 relevance check: {relevance_judgment['reason']}",
                "refined_query": query,
            }
            llm_trace = relevance_judgment.get("llm_trace")
        elif relevance_judgment["available"] and relevance_judgment["relevant"]:
            decision = {
                "continue": False,
                "reason": f"llama3 relevance check: {relevance_judgment['reason']}",
                "refined_query": query,
            }
            llm_trace = relevance_judgment.get("llm_trace")
        else:
            model = str(self.node.data.get("model", settings.chat_model))
            recent = json.dumps(tool_results[-5:], ensure_ascii=True)
            retrieval_quality_json = json.dumps(retrieval_quality, ensure_ascii=True)
            prompt = (
                "You are a reflection controller.\n"
                "Decide if current tool outputs are sufficient to answer the query.\n"
                "Prioritize retrieved content quality and evidence coverage, not tool names.\n"
                "Use retrieval_quality as a strong signal but not an absolute gate.\n"
                "If docs are totally empty, continue should be true. If docs are moderate and this is not first retry, continue can be false.\n"
                'Return strict JSON: {"continue":true|false,"reason":"...","refined_query":"..."}\n'
                f"Query: {query}\n"
                f"Current loop: {loop_count} / {max_loops}\n"
                f"Recent tool outputs: {recent}\n"
                f"Retrieval quality signals: {retrieval_quality_json}"
            )
            raw = _ollama_chat(
                model=model,
                messages=[
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            parsed = _parse_json(raw)
            decision = {
                "continue": bool(parsed.get("continue", False)),
                "reason": str(parsed.get("reason", "")).strip() or "reflection decision",
                "refined_query": str(parsed.get("refined_query", query)).strip() or query,
            }
            llm_trace = {
                "provider": "ollama_chat",
                "model": model,
                "input": prompt,
                "output": raw,
            }

        if decision["continue"]:
            context.shared_state["query"] = decision["refined_query"]
            context.last_query = decision["refined_query"]
            context.shared_state["plan_cursor"] = 0
            context.shared_state["plan"] = {}
            context.shared_state["reflection_loops"] = loop_count + 1

        context.shared_state.setdefault("history", []).append({"node": "reflection", "decision": decision})
        return {
            "text": decision["reason"],
            **decision,
            "retrieval_quality": retrieval_quality,
            "relevance_judgment": relevance_judgment,
            "llm_trace": llm_trace,
        }


class FinalAnswerNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        query = str(context.shared_state.get("query", context.last_query)).strip()
        if not query:
            raise HTTPException(status_code=400, detail=f"FinalAnswerNode '{self.node.id}' requires query")

        sources = context.last_sources or _latest_retrieved_sources(context.shared_state)
        model = self.node.data.get("model")
        model_name = str(model) if model else None
        if sources:
            answer = generate_answer(query=query, sources=sources, model=model_name)
            llm_trace = {
                "provider": "generate_answer",
                "model": model_name or settings.chat_model,
                "input": {
                    "query": query,
                    "source_count": len(sources),
                    "source_preview": [source.content[:280] for source in sources[:3]],
                },
                "output": answer,
            }
        else:
            tool_results = context.shared_state.get("tool_results", [])
            memory_text = "\n".join(
                f"{entry.get('tool')}: {entry.get('output')}" for entry in tool_results[-8:]
            )
            prompt = (
                "Produce a concise final answer using available tool outputs.\n"
                f"User query: {query}\n"
                f"Tool outputs:\n{memory_text or '(none)'}"
            )
            answer = _ollama_chat(
                model=model_name or settings.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            llm_trace = {
                "provider": "ollama_chat",
                "model": model_name or settings.chat_model,
                "input": prompt,
                "output": answer,
            }

        context.shared_state["final_answer"] = answer
        context.shared_state.setdefault("history", []).append({"node": "final_answer"})
        return {
            "text": answer,
            "query": query,
            "sources": [source.model_dump() for source in sources],
            "llm_trace": llm_trace,
        }


def build_node_executor(node: WorkflowNode) -> BaseWorkflowNode:
    mapping = {
        "InputNode": InputNode,
        "RetrieverNode": RetrieverNode,
        "LLMNode": LLMNode,
        "AgentNode": AgentNode,
        "PlannerNode": PlannerNode,
        "ToolSelectorNode": ToolSelectorNode,
        "ToolExecutorNode": ToolExecutorNode,
        "ReflectionNode": ReflectionNode,
        "FinalAnswerNode": FinalAnswerNode,
        "OutputNode": OutputNode,
    }
    executor_cls = mapping.get(node.type)
    if not executor_cls:
        raise HTTPException(status_code=400, detail=f"Unsupported node type: {node.type}")
    return executor_cls(node)


def _ollama_chat(
    model: str, messages: list[dict[str, str]], temperature: float
) -> str:
    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("message", {}).get("content", "")
    except requests.RequestException as exc:
        raise HTTPException(status_code=500, detail=f"Agent LLM error: {exc}") from exc


def _parse_json(raw: str) -> dict[str, Any]:
    candidate = raw.strip()
    if not candidate:
        return {}
    if not candidate.startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def _safe_calculate(expression: str) -> float:
    allowed = set("0123456789+-*/(). ")
    if not expression or any(ch not in allowed for ch in expression):
        raise HTTPException(status_code=400, detail="Calculator received invalid expression")
    try:
        return float(eval(expression, {"__builtins__": {}}, {}))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Calculator error: {exc}") from exc


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "null"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _web_search(query: str) -> str:
    try:
        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=10,
        )
        response.raise_for_status()
        text = response.text
        return text[:1200]
    except requests.RequestException:
        return "Web search unavailable."


def _assess_retrieval_quality(query: str, tool_results: list[dict[str, Any]]) -> dict[str, Any]:
    retrieve_entries = [
        item for item in tool_results if str(item.get("tool", "")).strip().lower() == "retrieve"
    ]
    if not retrieve_entries:
        return {
            "has_retrieve_call": False,
            "doc_count": 0,
            "non_empty_docs": 0,
            "avg_doc_length": 0,
            "query_overlap_ratio": 0.0,
            "is_sufficient": False,
        }

    latest = retrieve_entries[-1]
    docs = latest.get("output", [])
    if not isinstance(docs, list):
        docs = [str(docs)]
    text_docs = [str(doc).strip() for doc in docs]
    non_empty_docs = [doc for doc in text_docs if doc]
    avg_doc_length = (
        int(sum(len(doc) for doc in non_empty_docs) / len(non_empty_docs))
        if non_empty_docs
        else 0
    )

    query_tokens = {token for token in query.lower().split() if len(token) > 2}
    all_docs_text = " ".join(non_empty_docs).lower()
    overlap_count = sum(1 for token in query_tokens if token in all_docs_text)
    overlap_ratio = round(overlap_count / max(1, len(query_tokens)), 3)

    is_sufficient = len(non_empty_docs) >= 2 and avg_doc_length >= 60 and overlap_ratio >= 0.3

    return {
        "has_retrieve_call": True,
        "doc_count": len(text_docs),
        "non_empty_docs": len(non_empty_docs),
        "avg_doc_length": avg_doc_length,
        "query_overlap_ratio": overlap_ratio,
        "is_sufficient": is_sufficient,
    }


def _judge_retrieval_relevance(
    query: str, tool_results: list[dict[str, Any]], model: str = "llama3.2:latest"
) -> dict[str, Any]:
    retrieve_entries = [
        item for item in tool_results if str(item.get("tool", "")).strip().lower() == "retrieve"
    ]
    if not retrieve_entries:
        return {"available": False, "relevant": False, "reason": "no retrieve tool output"}

    latest = retrieve_entries[-1]
    docs = latest.get("output", [])
    if not isinstance(docs, list):
        docs = [str(docs)]
    normalized_docs = [str(doc).strip() for doc in docs if str(doc).strip()]
    if not normalized_docs:
        return {"available": True, "relevant": False, "reason": "no non-empty retrieved docs"}

    docs_block = "\n\n".join(
        f"[DOC {idx + 1}]\n{doc[:1200]}" for idx, doc in enumerate(normalized_docs[:5])
    )
    prompt = (
        "You are a strict relevance judge for retrieval quality.\n"
        "Given a user question and retrieved docs, decide if docs are relevant enough to answer.\n"
        'Return strict JSON only: {"relevant": true|false, "reason": "..."}\n'
        "Use false when docs are off-topic, too vague, or missing key evidence.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved docs:\n{docs_block}"
    )
    raw = _ollama_chat(
        model=model,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    parsed = _parse_json(raw)
    return {
        "available": True,
        "relevant": bool(parsed.get("relevant", False)),
        "reason": str(parsed.get("reason", "")).strip() or "llama3 relevance judgment",
        "llm_trace": {
            "provider": "ollama_chat",
            "model": model,
            "input": prompt,
            "output": raw,
        },
    }


def _latest_retrieved_sources(shared_state: dict[str, Any]) -> list[SourceChunk]:
    tool_results = shared_state.get("tool_results", [])
    if not isinstance(tool_results, list):
        return []
    for item in reversed(tool_results):
        if str(item.get("tool", "")).strip().lower() != "retrieve":
            continue
        raw_sources = item.get("sources", [])
        if not isinstance(raw_sources, list):
            continue
        parsed_sources: list[SourceChunk] = []
        for raw in raw_sources:
            try:
                parsed_sources.append(SourceChunk.model_validate(raw))
            except Exception:  # noqa: BLE001
                continue
        if parsed_sources:
            return parsed_sources
    return []
