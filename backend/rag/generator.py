"""
LLM answer generation with structured output and source attribution.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import settings


class Generator:
    """Generates grounded answers from retrieved context using the LLM."""

    SYSTEM_PROMPT = """You are a helpful AI Knowledge Assistant. Answer the user's 
question based ONLY on the provided context. Follow these rules strictly:

1. Base your answer ONLY on the provided context chunks.
2. If the context does not contain enough information, say "I don't have enough 
   information in the available documents to answer this question fully."
3. When referencing information, cite the source using [Source N] notation.
4. Be concise but thorough.
5. Use markdown formatting for readability.
6. If multiple sources agree, synthesize the information.
7. If sources conflict, mention the discrepancy."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a grounded answer with source citations.
        Returns dict with 'answer' and 'confidence'.
        """
        if not chunks:
            return {
                "answer": "I don't have any documents to search through. "
                "Please upload some documents first.",
                "confidence": 0.0,
            }

        # Offline-safe fallback when no OpenAI key is configured.
        if not settings.OPENAI_API_KEY:
            top = chunks[:3]
            lines = [
                "I can't call the LLM right now (no `OPENAI_API_KEY` configured).",
                "",
                "Here are the most relevant excerpts I found:",
            ]
            for i, c in enumerate(top, start=1):
                snippet = (c.get("text") or "").strip()
                snippet = snippet[:500] + ("…" if len(snippet) > 500 else "")
                lines.append(f"- [Source {i}] {snippet}")
            answer = "\n".join(lines)
            avg_relevance = sum(c.get("relevance_score", 0.5) for c in top) / max(1, len(top))
            return {"answer": answer, "confidence": round(float(avg_relevance), 3)}

        # Build context block
        context_parts = []
        for i, chunk in enumerate(chunks):
            doc_name = chunk.get("metadata", {}).get("document_name", "Unknown")
            page = chunk.get("metadata", {}).get("page_number", "?")
            context_parts.append(
                f"[Source {i + 1}] (Document: {doc_name}, Page: {page})\n{chunk['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # Build messages
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Add conversation history for multi-turn context
        if conversation_history:
            for msg in conversation_history[-settings.MAX_CONVERSATION_HISTORY :]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        user_message = f"""Context:
{context}

Question: {query}

Please answer the question based on the context above. Cite sources using [Source N]."""

        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1500,
            )
            answer = response.choices[0].message.content.strip()

            # Estimate confidence from chunk relevance scores
            avg_relevance = sum(
                c.get("relevance_score", 0.5) for c in chunks
            ) / len(chunks)

            return {"answer": answer, "confidence": round(avg_relevance, 3)}

        except Exception as e:
            return {
                "answer": f"I encountered an error generating the answer: {str(e)}",
                "confidence": 0.0,
            }
