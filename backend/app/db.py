from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_pgvector_extension() -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def ensure_schema_updates() -> None:
    """Apply lightweight, idempotent schema updates for existing dev databases."""
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                ALTER TABLE evaluations
                ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(64)
                """
            )
        )
        connection.execute(
            text(
                """
                UPDATE evaluations
                SET conversation_id = 'legacy'
                WHERE conversation_id IS NULL OR conversation_id = ''
                """
            )
        )
        connection.execute(
            text(
                """
                ALTER TABLE evaluations
                ALTER COLUMN conversation_id SET NOT NULL
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_evaluations_conversation_id
                ON evaluations (conversation_id)
                """
            )
        )
