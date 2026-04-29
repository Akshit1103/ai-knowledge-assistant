"""
Lightweight vector store for semantic search.

Why not ChromaDB here?
- On some Windows setups (especially fresh dev boxes), Chroma's default embedding
  stack can import `onnxruntime` and fail due to missing native DLL deps.
- This project already computes embeddings via OpenAI; we can store them directly.

This module implements a small persisted store with the same interface the rest
of the codebase expects (`add_documents`, `search`, `delete_document`, etc.).
"""

import os
import pickle
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from openai import OpenAI

from config import settings


class VectorStore:
    """Manages document embeddings in ChromaDB for semantic retrieval."""

    def __init__(self):
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        self._path = os.path.join(settings.CHROMA_PERSIST_DIR, "vector_store.pkl")
        self._items: Dict[str, Dict[str, Any]] = {}  # chunk_id -> {text, metadata, embedding}
        self._load()
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API if configured, else fallback."""
        if settings.OPENAI_API_KEY:
            response = self.openai_client.embeddings.create(
                model=settings.EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        return [self._fallback_embedding(t) for t in texts]

    def add_documents(
        self,
        chunk_ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ):
        """Add document chunks with embeddings to the collection."""
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = chunk_ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            embeddings = self.get_embeddings(batch_texts)
            for cid, text, meta, emb in zip(batch_ids, batch_texts, batch_meta, embeddings):
                self._items[cid] = {"text": text, "metadata": meta, "embedding": emb}
        self._save()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Semantic search returning ranked results with cosine similarity."""
        if not self._items:
            return []

        q_emb = np.asarray(self.get_embeddings([query])[0], dtype=np.float32)
        q_emb = self._l2_normalize(q_emb)

        scored: List[Tuple[str, float]] = []
        for cid, item in self._items.items():
            emb = np.asarray(item["embedding"], dtype=np.float32)
            emb = self._l2_normalize(emb)
            score = float(np.dot(q_emb, emb))
            scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for cid, score in scored[: max(1, int(top_k))]:
            item = self._items[cid]
            out.append(
                {
                    "chunk_id": cid,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": score,
                }
            )
        return out

    def delete_document(self, document_name: str):
        """Remove all chunks belonging to a specific document."""
        to_delete = [
            cid
            for cid, item in self._items.items()
            if item.get("metadata", {}).get("document_name") == document_name
        ]
        for cid in to_delete:
            self._items.pop(cid, None)
        if to_delete:
            self._save()

    def get_document_count(self) -> int:
        return len(self._items)

    def get_all_document_names(self) -> List[str]:
        """Return unique document names stored in the collection."""
        doc_names = set()
        for item in self._items.values():
            meta = item.get("metadata") or {}
            if "document_name" in meta:
                doc_names.add(meta["document_name"])
        return sorted(doc_names)

    def get_chunks_for_document(self, document_name: str) -> int:
        """Return the number of chunks for a given document."""
        return sum(
            1
            for item in self._items.values()
            if (item.get("metadata") or {}).get("document_name") == document_name
        )

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                self._items = data
        except Exception:
            # Corrupt cache shouldn't brick the app.
            self._items = {}

    def _save(self):
        tmp = self._path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._items, f)
        os.replace(tmp, self._path)

    def _fallback_embedding(self, text: str, dim: int = 384) -> List[float]:
        """
        A deterministic, local embedding for dev/offline mode.
        Not comparable to real embeddings, but good enough to keep the system runnable.
        """
        vec = np.zeros((dim,), dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if not tokens:
            return vec.tolist()
        for t in tokens[:2048]:
            h = hash(t)
            idx = h % dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx] += sign
        vec = self._l2_normalize(vec)
        return vec.tolist()

    def _l2_normalize(self, v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 0:
            return v
        return v / n
