from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.schemas import SourceChunk
from app.services.embeddings import embed_query


def retrieve_similar_chunks(db: Session, query: str, k: int) -> list[SourceChunk]:
    query_embedding = embed_query(query)
    score_expr = Chunk.embedding.cosine_distance(query_embedding).label("score")
    stmt: Select = (
        select(Chunk, Document, score_expr)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(score_expr.asc())
        .limit(k)
    )
    rows = db.execute(stmt).all()

    sources: list[SourceChunk] = []
    for chunk, document, score in rows:
        sources.append(
            SourceChunk(
                document_id=document.id,
                document_name=document.name,
                chunk_id=chunk.id,
                content=chunk.content,
                metadata=chunk.chunk_metadata,
                score=float(score),
            )
        )
    return sources
