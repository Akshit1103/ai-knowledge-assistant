"""
Hybrid retrieval combining semantic (ChromaDB) and keyword (BM25) search
with Reciprocal Rank Fusion (RRF) for score merging.
"""

from typing import List, Dict, Any

from config import settings
from store.vector_store import VectorStore
from store.bm25_store import BM25Store


class HybridRetriever:
    """Fuses semantic and keyword search results using RRF."""

    RRF_K = 60  # Standard RRF constant

    def __init__(self, vector_store: VectorStore, bm25_store: BM25Store):
        self.vector_store = vector_store
        self.bm25_store = bm25_store

    def retrieve(
        self,
        queries: List[str],
        semantic_top_k: int = None,
        bm25_top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Run all queries through both stores and fuse with RRF.
        Returns deduplicated, ranked results.
        """
        sem_k = semantic_top_k or settings.SEMANTIC_TOP_K
        bm_k = bm25_top_k or settings.BM25_TOP_K

        # Collect per-query rankings
        semantic_rankings: List[List[Dict[str, Any]]] = []
        bm25_rankings: List[List[Dict[str, Any]]] = []

        for q in queries:
            semantic_rankings.append(self.vector_store.search(q, top_k=sem_k))
            bm25_rankings.append(self.bm25_store.search(q, top_k=bm_k))

        # Build RRF scores keyed by chunk_id
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}
        chunk_methods: Dict[str, set] = {}

        for ranking in semantic_rankings:
            for rank, result in enumerate(ranking):
                cid = result["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.RRF_K + rank + 1)
                chunk_data[cid] = result
                chunk_methods.setdefault(cid, set()).add("semantic")

        for ranking in bm25_rankings:
            for rank, result in enumerate(ranking):
                cid = result["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (self.RRF_K + rank + 1)
                chunk_data[cid] = result
                chunk_methods.setdefault(cid, set()).add("bm25")

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for cid in sorted_ids:
            methods = chunk_methods.get(cid, set())
            if len(methods) > 1:
                method_label = "hybrid"
            else:
                method_label = next(iter(methods))

            results.append(
                {
                    "chunk_id": cid,
                    "text": chunk_data[cid]["text"],
                    "metadata": chunk_data[cid]["metadata"],
                    "rrf_score": rrf_scores[cid],
                    "retrieval_method": method_label,
                }
            )

        return results
