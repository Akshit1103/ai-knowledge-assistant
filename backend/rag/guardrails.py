"""
Input/output guardrails and fallback logic.
"""

from typing import Dict, Any, Optional
from openai import OpenAI

from config import settings


class Guardrails:
    """Validates inputs and outputs, applies fallback when confidence is low."""

    BLOCKED_TOPICS = [
        "generate malware",
        "how to hack",
        "illegal activities",
        "create a virus",
        "bypass security",
    ]

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # ------------------------------------------------------------------
    # Input guardrails
    # ------------------------------------------------------------------

    def check_input(self, query: str) -> Dict[str, Any]:
        """
        Validate user input. Returns:
          {"allowed": bool, "reason": str | None}
        """
        # Length check
        if len(query.strip()) == 0:
            return {"allowed": False, "reason": "Query cannot be empty."}

        if len(query) > settings.GUARDRAIL_MAX_INPUT_LENGTH:
            return {
                "allowed": False,
                "reason": f"Query exceeds maximum length of {settings.GUARDRAIL_MAX_INPUT_LENGTH} characters.",
            }

        # Blocked-topic check (simple keyword scan)
        query_lower = query.lower()
        for topic in self.BLOCKED_TOPICS:
            if topic in query_lower:
                return {
                    "allowed": False,
                    "reason": "This query touches on a restricted topic and cannot be processed.",
                }

        return {"allowed": True, "reason": None}

    # ------------------------------------------------------------------
    # Output guardrails
    # ------------------------------------------------------------------

    def check_output(
        self,
        answer: str,
        confidence: float,
        chunks: list,
    ) -> Dict[str, Any]:
        """
        Validate the generated answer. Returns:
          {"answer": str, "is_fallback": bool, "confidence": float}
        """
        # No chunks at all → definite fallback
        if not chunks:
            return {
                "answer": "I don't have enough information in the available documents "
                "to answer this question. Please try uploading relevant documents first.",
                "is_fallback": True,
                "confidence": 0.0,
            }

        # Low confidence → qualify the answer
        if confidence < settings.CONFIDENCE_THRESHOLD:
            qualified = (
                "⚠️ **Low confidence answer** — the available documents may not "
                "contain sufficient information on this topic.\n\n" + answer
            )
            return {
                "answer": qualified,
                "is_fallback": True,
                "confidence": confidence,
            }

        return {
            "answer": answer,
            "is_fallback": False,
            "confidence": confidence,
        }
