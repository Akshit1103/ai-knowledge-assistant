"""
BM25-backed keyword store for lexical retrieval.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi

from config import settings


class BM25Store:
    """Manages a BM25 index for keyword-based retrieval with persistence."""

    def __init__(self):
        os.makedirs(settings.BM25_PERSIST_DIR, exist_ok=True)
        self.docs_path = os.path.join(settings.BM25_PERSIST_DIR, "bm25_docs.json")
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())

    def _load(self):
        """Load persisted documents and rebuild the BM25 index."""
        if os.path.exists(self.docs_path):
            with open(self.docs_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            if self.documents:
                tokenized = [self._tokenize(doc["text"]) for doc in self.documents]
                self.bm25 = BM25Okapi(tokenized)

    def _save(self):
        with open(self.docs_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def _rebuild_index(self):
        if self.documents:
            tokenized = [self._tokenize(doc["text"]) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add chunks to the BM25 index."""
        for chunk_id, text, meta in zip(chunk_ids, texts, metadatas):
            self.documents.append(
                {"chunk_id": chunk_id, "text": text, "metadata": meta}
            )
        self._rebuild_index()
        self._save()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Keyword search returning BM25-scored results."""
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "chunk_id": self.documents[idx]["chunk_id"],
                        "text": self.documents[idx]["text"],
                        "metadata": self.documents[idx]["metadata"],
                        "score": float(scores[idx]),
                    }
                )
        return results

    def delete_document(self, document_name: str):
        """Remove all chunks for a document and rebuild the index."""
        self.documents = [
            doc
            for doc in self.documents
            if doc["metadata"].get("document_name") != document_name
        ]
        self._rebuild_index()
        self._save()

    def get_document_count(self) -> int:
        return len(self.documents)
