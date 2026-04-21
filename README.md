# RAG Flow (FastAPI + Next.js)

Minimal Retrieval-Augmented Generation stack with:
- File upload (`PDF`, `DOCX`, `TXT`)
- Chunking + Ollama embeddings
- PostgreSQL + `pgvector` storage
- Similarity retrieval (`top-k`)
- Chat API and frontend chat UI with source citations

## Project Structure

- `backend/` - FastAPI RAG backend
- `frontend/` - Next.js minimal UI

## Backend Setup (FastAPI)

### 1) Start infrastructure with Docker Compose

Start PostgreSQL (`pgvector`) only:

```bash
docker compose up -d
```

Check status:

```bash
docker compose ps
```

### 2) Install backend dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Configure environment

```bash
cp .env.example .env
```

Set:
- `OLLAMA_BASE_URL` (defaults to `http://localhost:11434`)
- `DATABASE_URL` (default already points to local docker)
- `EMBEDDING_MODEL` (default `nomic-embed-text:latest`)
- `CHAT_MODEL` (default `qwen3:8b`)
- `EMBEDDING_DIMENSION` (default `768`)

Pull local Ollama models:

```bash
ollama pull nomic-embed-text:latest
ollama pull qwen3:8b
```

### 4) Run API server

```bash
uvicorn app.main:app --reload --port 8000
```

Backend endpoints:
- `GET /api/health`
- `POST /api/upload` (multipart `file`)
- `POST /api/chat` (`{"query":"...", "k":5}`)

## Frontend Setup (Next.js)

### 1) Install dependencies

```bash
cd frontend
npm install
```

### 2) Configure API URL

```bash
cp .env.local.example .env.local
```

Defaults to `http://localhost:8000/api`.

### 3) Run frontend

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Database Schema

### `documents`
- `id` (PK)
- `name` (string)

### `chunks`
- `id` (PK)
- `document_id` (FK -> `documents.id`)
- `content` (text)
- `embedding` (`vector(768)` by default, configurable)
- `metadata` (`json`)

## RAG Flow

1. Upload file on frontend (`/upload`)
2. Backend extracts text (PyMuPDF for PDF, python-docx for DOCX)
3. Text is chunked with `RecursiveCharacterTextSplitter`
4. Chunks are embedded with Ollama (`nomic-embed-text:latest`)
5. Chunks + vectors are stored in Postgres `pgvector`
6. Chat query (`/chat`) embeds the query and retrieves top-k similar chunks
7. Retrieved context is injected into Ollama LLM prompt (`qwen3:8b`)
8. API returns:
   - `answer`
   - `sources` (chunk-level metadata for UI rendering)
