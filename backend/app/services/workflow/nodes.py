from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy.orm import Session

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
        if context.last_sources is None:
            raise HTTPException(
                status_code=400,
                detail=f"LLMNode '{self.node.id}' requires RetrieverNode output",
            )

        enriched_query = f"{template}\n\nQuestion: {query}"
        model = self.node.data.get("model")
        model_name = str(model) if model else None
        answer = generate_answer(
            query=enriched_query,
            sources=context.last_sources,
            model=model_name,
        )
        return {
            "text": answer,
            "sources": [source.model_dump() for source in context.last_sources],
        }


class OutputNode(BaseWorkflowNode):
    def execute(self, context: WorkflowContext) -> dict:
        if not context.incoming_text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"OutputNode '{self.node.id}' has no input",
            )
        return {"text": context.incoming_text}


def build_node_executor(node: WorkflowNode) -> BaseWorkflowNode:
    mapping = {
        "InputNode": InputNode,
        "RetrieverNode": RetrieverNode,
        "LLMNode": LLMNode,
        "OutputNode": OutputNode,
    }
    executor_cls = mapping.get(node.type)
    if not executor_cls:
        raise HTTPException(status_code=400, detail=f"Unsupported node type: {node.type}")
    return executor_cls(node)
