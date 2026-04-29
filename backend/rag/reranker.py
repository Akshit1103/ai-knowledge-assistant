"""
LLM-based re-ranking of retrieved chunks for relevance.
"""

import json
from typing import List, Dict, Any
from openai import OpenAI

from config import settings


class Reranker:
    """Scores each chunk for relevance using the LLM and re-orders them."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Score each chunk 0-10 for relevance, return the top_k highest.
        """
        k = top_k or settings.RERANK_TOP_K

        if not chunks:
            return []

        # Build numbered list of chunks for the prompt
        chunk_list = ""
        for i, chunk in enumerate(chunks):
            snippet = chunk["text"][:300].replace("\n", " ")
            chunk_list += f"[{i}] {snippet}\n"

        prompt = f"""You are a relevance scoring expert. Given a user query and a list 
of text chunks, score each chunk from 0 to 10 for how relevant it is to answering the query.

Query: "{query}"

Chunks:
{chunk_list}

Return a JSON array of objects with "index" (int) and "score" (float 0-10).
Only return the JSON array, no other text."""

        try:
            response = self.client.chat.completions.create(
                model=settings.RERANKER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            scores = json.loads(content)

            # Attach scores to chunks
            score_map: Dict[int, float] = {}
            for item in scores:
                idx = int(item["index"])
                score_map[idx] = float(item["score"])

            scored_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_copy = dict(chunk)
                chunk_copy["relevance_score"] = score_map.get(i, 0.0) / 10.0
                scored_chunks.append(chunk_copy)

            # Sort by relevance_score descending
            scored_chunks.sort(key=lambda c: c["relevance_score"], reverse=True)
            return scored_chunks[:k]

        except Exception:
            # Fallback: keep original order, assign descending scores
            result = []
            for i, chunk in enumerate(chunks[:k]):
                chunk_copy = dict(chunk)
                chunk_copy["relevance_score"] = max(0.1, 1.0 - i * 0.1)
                result.append(chunk_copy)
            return result
