from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.config import settings


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    files: Mapped[list["DataFile"]] = relationship(
        "DataFile",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    graph_entities: Mapped[list["GraphEntity"]] = relationship(
        "GraphEntity",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    graph_relations: Mapped[list["GraphRelation"]] = relationship(
        "GraphRelation",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class DataFile(Base):
    __tablename__ = "files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    file_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="files")
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="file",
        cascade="all, delete-orphan",
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_id: Mapped[int] = mapped_column(
        ForeignKey("files.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.embedding_dimension), nullable=False
    )
    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    file: Mapped[DataFile] = relationship("DataFile", back_populates="chunks")
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="chunks")
    graph_links: Mapped[list["ChunkEntityLink"]] = relationship(
        "ChunkEntityLink",
        back_populates="chunk",
        cascade="all, delete-orphan",
    )


class GraphEntity(Base):
    __tablename__ = "graph_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False, default="concept")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    aliases: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    entity_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(settings.embedding_dimension), nullable=True)

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="graph_entities")
    outgoing_relations: Mapped[list["GraphRelation"]] = relationship(
        "GraphRelation",
        back_populates="source_entity",
        foreign_keys="GraphRelation.source_entity_id",
        cascade="all, delete-orphan",
    )
    incoming_relations: Mapped[list["GraphRelation"]] = relationship(
        "GraphRelation",
        back_populates="target_entity",
        foreign_keys="GraphRelation.target_entity_id",
        cascade="all, delete-orphan",
    )
    chunk_links: Mapped[list["ChunkEntityLink"]] = relationship(
        "ChunkEntityLink",
        back_populates="entity",
        cascade="all, delete-orphan",
    )


class GraphRelation(Base):
    __tablename__ = "graph_relations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    source_entity_id: Mapped[int] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    target_entity_id: Mapped[int] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    relation: Mapped[str] = mapped_column(String(128), nullable=False, default="related_to")
    weight: Mapped[float] = mapped_column(nullable=False, default=1.0)
    evidence_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    relation_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="graph_relations")
    source_entity: Mapped[GraphEntity] = relationship(
        "GraphEntity",
        back_populates="outgoing_relations",
        foreign_keys=[source_entity_id],
    )
    target_entity: Mapped[GraphEntity] = relationship(
        "GraphEntity",
        back_populates="incoming_relations",
        foreign_keys=[target_entity_id],
    )


class ChunkEntityLink(Base):
    __tablename__ = "chunk_entity_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    chunk_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(nullable=False, default=0.5)
    link_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=False, default=dict)

    chunk: Mapped[Chunk] = relationship("Chunk", back_populates="graph_links")
    entity: Mapped[GraphEntity] = relationship("GraphEntity", back_populates="chunk_links")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    scores: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
