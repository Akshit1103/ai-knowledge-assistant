"""
Orchestrates document loading, chunking, and indexing into both stores.
"""

import os
import uuid
from typing import Dict, Any

from config import settings
from ingestion.document_loader import DocumentLoader
from ingestion.chunker import TextChunker
from store.vector_store import VectorStore
from store.bm25_store import BM25Store


class Indexer:
    """End-to-end indexing: load → chunk → store in ChromaDB + BM25."""

    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.chunker = TextChunker()

    def index_file(self, file_path: str) -> Dict[str, Any]:
        """
        Index a single file. Returns metadata about the indexed document.
        """
        document = DocumentLoader.load(file_path)
        chunks = self.chunker.chunk_document(document)

        if not chunks:
            raise ValueError("No content could be extracted from the document.")

        doc_id = str(uuid.uuid4())
        chunk_ids = [f"{doc_id}_chunk_{c['chunk_index']}" for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "document_name": c["document_name"],
                "document_id": doc_id,
                "page_number": c["page_number"] or 0,
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]

        # Index into both stores
        self.vector_store.add_documents(chunk_ids, texts, metadatas)
        self.bm25_store.add_documents(chunk_ids, texts, metadatas)

        file_size = os.path.getsize(file_path)

        return {
            "document_id": doc_id,
            "name": document["name"],
            "num_chunks": len(chunks),
            "size_bytes": file_size,
        }

    def delete_document(self, document_name: str):
        """Remove a document from both stores."""
        self.vector_store.delete_document(document_name)
        self.bm25_store.delete_document(document_name)
