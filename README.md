# AI Knowledge Assistant (Advanced RAG System)

A full-stack **Retrieval-Augmented Generation (RAG)** app that answers user questions over uploaded documents using a production-style pipeline:

**User Query → Query Expansion → Hybrid Retrieval (Semantic + BM25) → Re-ranking → Answer Generation → Guardrails + Evaluation → Structured Answer with Sources**

## What’s included

- **Hybrid retrieval**: semantic retrieval + keyword (BM25) retrieval fused with **Reciprocal Rank Fusion (RRF)**.
- **Re-ranking**: LLM-based re-ranking (with a fallback when no OpenAI key is configured).
- **Multi-query expansion**: generates multiple query variants to improve recall (LLM-based with deterministic fallback).
- **Conversation memory**: per-session conversation history is used for multi-turn context.
- **Evaluation**: LLM-as-judge scoring for **faithfulness** and **relevance** (optional per request).
- **Guardrails + fallback**:
  - input checks (empty, too long, restricted topics)
  - output fallback for low-confidence answers
- **Structured outputs**: API returns an `answer`, `sources[]`, `pipeline_steps[]`, `confidence`, and `conversation_id`.

## Repo structure

- `backend/`: FastAPI app + ingestion + RAG pipeline
  - `backend/main.py`: API server (`/api/upload`, `/api/query`, `/api/documents`, `/api/conversations/...`)
  - `backend/ingestion/`: PDF/text loaders, chunking, indexing
  - `backend/store/`: vector store + BM25 store
  - `backend/rag/`: query expansion, hybrid retrieval, re-ranking, generator, guardrails, evaluation, pipeline orchestrator
- `frontend/`: React + Vite + Tailwind UI
  - upload documents, run chat queries, view sources/evaluation/pipeline steps
- `documents/`: sample documents to test with

## Running locally

### Backend (Windows / PowerShell)

The backend is designed to run well on Windows using a **uv-managed Python 3.11** environment.

1) Install dependencies into `backend/.venv` (uv)

```powershell
python -m pip install -U uv
python -m uv python install 3.11
python -m uv venv --python 3.11 backend\.venv
python -m uv pip install -r backend\requirements.txt --python backend\.venv
```

2) Start the API

```powershell
cd backend
.\.venv\Scripts\python -m uvicorn main:app --reload --port 8000
```

Open Swagger docs at `http://localhost:8000/docs`.

### Frontend

```powershell
cd frontend
bun install
bun dev
```

The UI calls the backend at `http://localhost:8000` by default.

If you want a custom API base, set `VITE_API_BASE` when running the dev server:

```powershell
$env:VITE_API_BASE="http://localhost:8000"
bun dev
```

## OpenAI key: required?

**Not required to run the app**.

When `OPENAI_API_KEY` is **not** set:
- query expansion uses deterministic variants
- vector store uses deterministic local embeddings (so “semantic” retrieval is approximate)
- generator returns a safe fallback answer (top excerpts as sources)

When `OPENAI_API_KEY` **is** set, you get the full “advanced RAG” behavior:
- real OpenAI embeddings
- LLM re-ranking
- LLM grounded answer generation
- LLM evaluation (faithfulness/relevance)
- LLM query expansion

To enable OpenAI features, create `backend/.env`:

```env
OPENAI_API_KEY=your_key_here
```

Other useful knobs are in `backend/config.py` (e.g., `LLM_MODEL`, `RERANKER_MODEL`, `CHUNK_SIZE`, `SEMANTIC_TOP_K`, etc.).

## API overview

- `POST /api/upload`: upload and index a document (`.pdf`, `.txt`, `.md`)
- `POST /api/query`: run RAG (`conversation_id` optional; `include_evaluation` optional)
- `GET /api/documents`: list indexed documents
- `DELETE /api/documents/{document_name}`: remove document from index
- `GET /api/conversations/{session_id}`: fetch session history
- `DELETE /api/conversations/{session_id}`: clear session history
- `GET /api/health`: basic health + counts

## Production checklist (what’s left)

This repo is a strong **intermediate** implementation. To make it production-ready for “upload any PDF and chat with it”, here’s what you’d typically add:

### Document ingestion & parsing
- **Robust PDF extraction**: handle scanned PDFs (OCR), complex layouts, tables, images.
  - Add OCR (e.g., Tesseract) and a layout-aware parser when `extract_text()` returns poor results.
- **Async/background indexing**: upload returns a job id; indexing runs in a worker queue (Celery/RQ/Sidekiq/etc.).
- **Deduplication**: hash documents and chunks; prevent duplicate indexing.
- **Metadata & provenance**: persist upload time, user, doc id, page refs, and chunk offsets.

### Storage & scale
- **Replace local persistence**:
  - current vector store persistence is local disk (pickle) for portability/dev
  - move to a production vector DB (Postgres + pgvector, Qdrant, Pinecone, Weaviate, etc.)
- **Multi-tenant isolation**: separate indices/collections per workspace/user.
- **Backups & migrations**: versioned schema for stored chunks and embeddings.

### Security & access control
- **Authentication/authorization** (users, orgs, roles).
- **Upload security**: virus scanning, file size limits, content-type validation, sandbox parsing.
- **Prompt-injection defenses**: stronger policies, tool isolation, and “instruction hierarchy” handling.
- **PII handling**: redaction and policy-based filtering.

### Retrieval quality
- **Better chunking**: structure-aware chunking (headings/sections), semantic splitters.
- **Caching**: cache embeddings and retrieval results.
- **Re-ranking model choice**: use a dedicated reranker model (or cross-encoder) for cost/latency.
- **Answer citations**: optionally enforce “citation required per sentence” and validate citations.

### Reliability, cost, and observability
- **Rate limiting** + request quotas per user.
- **Timeouts** and retries for LLM calls; circuit breaker for provider outages.
- **Tracing/metrics/logs**: request ids, stage timings, token usage, error dashboards.
- **Evaluation harness**: offline evaluation sets + regression tests; store eval results over time.

### Deployment
- **Containerize** backend + worker + DB/vector DB.
- **Serve frontend builds** (CDN) and lock down CORS.
- **Secrets management** for `OPENAI_API_KEY` and other credentials.

