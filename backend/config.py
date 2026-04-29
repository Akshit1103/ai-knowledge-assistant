"""
Application configuration loaded from environment variables.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration for the AI Knowledge Assistant."""

    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "gpt-4o-mini")

    # Storage
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    BM25_PERSIST_DIR: str = os.getenv("BM25_PERSIST_DIR", "./data/bm25")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Retrieval
    SEMANTIC_TOP_K: int = int(os.getenv("SEMANTIC_TOP_K", "10"))
    BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "10"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))

    # Query Expansion
    NUM_QUERY_EXPANSIONS: int = int(os.getenv("NUM_QUERY_EXPANSIONS", "3"))

    # Conversation
    MAX_CONVERSATION_HISTORY: int = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))

    # Guardrails
    GUARDRAIL_MAX_INPUT_LENGTH: int = int(os.getenv("GUARDRAIL_MAX_INPUT_LENGTH", "2000"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))


settings = Settings()
