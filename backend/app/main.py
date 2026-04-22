from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.db import ensure_pgvector_extension, ensure_schema_updates, engine
from app.models import Base

app = FastAPI(title="RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    ensure_pgvector_extension()
    Base.metadata.create_all(bind=engine)
    ensure_schema_updates()


app.include_router(router, prefix="/api")
