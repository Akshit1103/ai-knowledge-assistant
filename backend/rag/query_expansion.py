"""
Multi-query expansion using LLM to generate alternative phrasings.
"""

import json
import re
from typing import List
from openai import OpenAI

from config import settings


class QueryExpander:
    """Generates multiple query variations to improve retrieval recall."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def expand(self, query: str, num_expansions: int = None) -> List[str]:
        """
        Given a user query, generate N alternative phrasings.
        Returns a list that always starts with the original query.
        """
        n = num_expansions or settings.NUM_QUERY_EXPANSIONS
        n = max(0, int(n))

        q = (query or "").strip()
        if not q:
            return [""]

        # If there's no API key configured, skip the network call entirely.
        if not settings.OPENAI_API_KEY:
            return [q] + self._fallback_expansions(q, n)

        prompt = f"""You are a search query expansion expert. Given the user's query, 
generate {n} alternative phrasings that capture the same intent but use different 
keywords, synonyms, or perspectives. This helps improve document retrieval.

Rules:
- Each alternative should emphasize different aspects or terminology
- Keep alternatives concise (1-2 sentences max)
- Do NOT answer the query — only rephrase it
- Return ONLY a JSON array of strings

User query: "{query}"

Return a JSON array of {n} alternative queries:"""

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            alternatives = json.loads(content)
            if isinstance(alternatives, list):
                cleaned = [self._squash_ws(str(a)) for a in alternatives if str(a).strip()]
                cleaned = [c for c in cleaned if c.lower() != q.lower()]
                if cleaned:
                    return [q] + cleaned[:n]
        except Exception:
            # Fall back to deterministic variants (helps both BM25 + semantic retrieval).
            return [q] + self._fallback_expansions(q, n)

        # Fallback: return only the original
        return [q] + self._fallback_expansions(q, n)

    def _fallback_expansions(self, query: str, n: int) -> List[str]:
        """
        Deterministic query variants for offline / failure cases.
        Goal: improve keyword recall for BM25 while keeping semantic intent.
        """
        if n <= 0:
            return []

        base = self._squash_ws(query)
        keywords = self._keywords(base)

        candidates: List[str] = []

        # Variant 1: keyword-only (BM25 friendly)
        if keywords:
            candidates.append(" ".join(keywords))

        # Variant 2: "explain/overview" variant (common doc phrasing)
        candidates.append(f"overview of {base}")

        # Variant 3: "key points" variant
        candidates.append(f"key points about {base}")

        # Variant 4: if query looks like a question, also try a statement form
        if base.endswith("?"):
            candidates.append(base[:-1].strip())

        # Dedupe and trim
        seen = {base.lower()}
        out: List[str] = []
        for c in candidates:
            c2 = self._squash_ws(c)
            if not c2:
                continue
            key = c2.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(c2)
            if len(out) >= n:
                break
        return out

    def _keywords(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        stop = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "but",
            "by",
            "can",
            "do",
            "does",
            "for",
            "from",
            "how",
            "i",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "the",
            "to",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "with",
            "you",
            "your",
        }
        keywords: List[str] = []
        for t in tokens:
            if len(t) < 3:
                continue
            if t in stop:
                continue
            if t not in keywords:
                keywords.append(t)
        return keywords[:12]

    def _squash_ws(self, s: str) -> str:
        return " ".join((s or "").strip().split())
