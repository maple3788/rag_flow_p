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
    top_k: int = int(os.getenv("TOP_K", "5"))
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))


settings = Settings()
