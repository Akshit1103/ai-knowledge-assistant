"""
LLM-as-judge evaluation for faithfulness and relevance.
"""

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import settings


class Evaluator:
    """Scores answers on faithfulness (grounded in context) and relevance (addresses the query)."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Returns faithfulness_score, relevance_score (0-1) with reasoning.
        """
        if not chunks or not answer:
            return {
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "faithfulness_reasoning": "No context or answer to evaluate.",
                "relevance_reasoning": "No context or answer to evaluate.",
            }

        context = "\n\n".join(c["text"] for c in chunks[:5])

        prompt = f"""You are an evaluation expert. Score the following answer on two dimensions.

**Query:** {query}

**Context (retrieved chunks):**
{context}

**Answer:** {answer}

Score each dimension from 0.0 to 1.0:

1. **Faithfulness**: Is the answer fully supported by the context? 
   1.0 = every claim is directly supported, 0.0 = entirely hallucinated.

2. **Relevance**: Does the answer address the user's query?
   1.0 = perfectly addresses the query, 0.0 = completely off-topic.

Return ONLY a JSON object:
{{
  "faithfulness_score": <float>,
  "faithfulness_reasoning": "<brief explanation>",
  "relevance_score": <float>,
  "relevance_reasoning": "<brief explanation>"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0]
            result = json.loads(content)

            return {
                "faithfulness_score": max(0.0, min(1.0, float(result.get("faithfulness_score", 0)))),
                "relevance_score": max(0.0, min(1.0, float(result.get("relevance_score", 0)))),
                "faithfulness_reasoning": str(result.get("faithfulness_reasoning", "")),
                "relevance_reasoning": str(result.get("relevance_reasoning", "")),
            }
        except Exception as e:
            return {
                "faithfulness_score": 0.0,
                "relevance_score": 0.0,
                "faithfulness_reasoning": f"Evaluation failed: {str(e)}",
                "relevance_reasoning": f"Evaluation failed: {str(e)}",
            }
