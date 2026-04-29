"""
FastAPI application — REST API for the AI Knowledge Assistant.
"""

import os
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from store.vector_store import VectorStore
from store.bm25_store import BM25Store
from ingestion.indexer import Indexer
from rag.pipeline import RAGPipeline
from models.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    DocumentListResponse,
    DocumentInfo,
    HealthResponse,
)

# ------------------------------------------------------------------ #
# Initialise core services
# ------------------------------------------------------------------ #
vector_store = VectorStore()
bm25_store = BM25Store()
indexer = Indexer(vector_store, bm25_store)
pipeline = RAGPipeline(vector_store, bm25_store)

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# FastAPI app
# ------------------------------------------------------------------ #
app = FastAPI(
    title="AI Knowledge Assistant",
    description="Production-grade RAG system with hybrid retrieval, re-ranking, "
    "query expansion, guardrails, and evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ------------------------------------------------------------------ #
# Endpoints
# ------------------------------------------------------------------ #


@app.get("/", include_in_schema=False)
async def root():
    """Redirect to the frontend."""
    from fastapi.responses import FileResponse

    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AI Knowledge Assistant API — visit /docs for Swagger UI."}


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """System health check."""
    return HealthResponse(
        status="healthy",
        total_documents=len(vector_store.get_all_document_names()),
        total_chunks=vector_store.get_document_count(),
    )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run a query through the full RAG pipeline."""
    result = pipeline.run(
        query=request.query,
        conversation_id=request.conversation_id,
        include_evaluation=request.include_evaluation,
    )
    return result


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document (PDF, TXT, MD)."""
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".pdf", ".txt", ".md", ".markdown"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .md",
        )

    # Save file
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = indexer.index_file(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")

    return UploadResponse(
        document_id=result["document_id"],
        name=result["name"],
        num_chunks=result["num_chunks"],
        message=f"Successfully indexed '{result['name']}' into {result['num_chunks']} chunks.",
    )


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    doc_names = vector_store.get_all_document_names()
    documents: List[DocumentInfo] = []

    for name in doc_names:
        file_path = os.path.join(settings.UPLOAD_DIR, name)
        size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        num_chunks = vector_store.get_chunks_for_document(name)
        documents.append(
            DocumentInfo(
                id=name,
                name=name,
                size_bytes=size,
                num_chunks=num_chunks,
                uploaded_at=datetime.utcnow(),
            )
        )

    return DocumentListResponse(
        documents=documents,
        total_chunks=vector_store.get_document_count(),
    )


@app.delete("/api/documents/{document_name}")
async def delete_document(document_name: str):
    """Remove a document from the index."""
    indexer.delete_document(document_name)
    file_path = os.path.join(settings.UPLOAD_DIR, document_name)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": f"Document '{document_name}' deleted."}


@app.get("/api/conversations/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history for a session."""
    history = pipeline.memory.get_history(session_id)
    return {"session_id": session_id, "messages": history}


@app.delete("/api/conversations/{session_id}")
async def clear_conversation(session_id: str):
    """Clear a conversation session."""
    pipeline.memory.clear_session(session_id)
    return {"message": f"Conversation '{session_id}' cleared."}


# ------------------------------------------------------------------ #
# Entry-point
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
