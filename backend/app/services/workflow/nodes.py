from dataclasses import dataclass
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


def build_node_executor(node: WorkflowNode) -> BaseWorkflowNode:
    mapping = {
        "InputNode": InputNode,
        "RetrieverNode": RetrieverNode,
        "LLMNode": LLMNode,
        "AgentNode": AgentNode,
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
