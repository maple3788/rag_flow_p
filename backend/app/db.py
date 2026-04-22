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
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    config JSON NOT NULL DEFAULT '{}'::json,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS files (
                    id SERIAL PRIMARY KEY,
                    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                    filename VARCHAR(512) NOT NULL,
                    raw_text TEXT NOT NULL,
                    metadata JSON NOT NULL DEFAULT '{}'::json
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_files_dataset_id
                ON files (dataset_id)
                """
            )
        )
        connection.execute(
            text(
                """
                ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS file_id INTEGER
                """
            )
        )
        connection.execute(
            text(
                """
                ALTER TABLE chunks
                ADD COLUMN IF NOT EXISTS dataset_id INTEGER
                """
            )
        )
        connection.execute(
            text(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'chunks'
                          AND column_name = 'document_id'
                    ) THEN
                        ALTER TABLE chunks
                        ALTER COLUMN document_id DROP NOT NULL;
                    END IF;
                END
                $$;
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO datasets (name, description, config)
                VALUES ('legacy', 'Auto-migrated legacy dataset', '{}'::json)
                ON CONFLICT (name) DO NOTHING
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO files (dataset_id, filename, raw_text, metadata)
                SELECT
                    (SELECT id FROM datasets WHERE name = 'legacy'),
                    COALESCE(d.name, 'legacy_document_' || d.id::text),
                    COALESCE(string_agg(c.content, E'\n\n' ORDER BY c.id), ''),
                    '{}'::json
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'chunks' AND column_name = 'document_id'
                )
                AND NOT EXISTS (
                    SELECT 1 FROM files f WHERE f.filename = d.name
                )
                GROUP BY d.id, d.name
                """
            )
        )
        connection.execute(
            text(
                """
                UPDATE chunks c
                SET dataset_id = f.dataset_id,
                    file_id = f.id
                FROM files f
                WHERE c.file_id IS NULL
                AND c.document_id IS NOT NULL
                AND f.filename = (
                    SELECT d.name FROM documents d WHERE d.id = c.document_id
                )
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_chunks_file_id
                ON chunks (file_id)
                """
            )
        )
        connection.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_chunks_dataset_id
                ON chunks (dataset_id)
                """
            )
        )
