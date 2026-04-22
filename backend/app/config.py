import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/rag_db",
    )
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen3:8b")
    chat_model_options: tuple[str, ...] = ("qwen3:8b", "llama3.2:latest")
    top_k: int = int(os.getenv("TOP_K", "5"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    elasticsearch_url: str = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
    elasticsearch_username: str | None = os.getenv("ELASTICSEARCH_USERNAME")
    elasticsearch_password: str | None = os.getenv("ELASTICSEARCH_PASSWORD")
    elasticsearch_timeout_seconds: int = int(os.getenv("ELASTICSEARCH_TIMEOUT_SECONDS", "10"))


settings = Settings()
