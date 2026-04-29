"""
Pydantic models for request/response schemas throughout the application.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationMessage(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    include_evaluation: bool = True


class Source(BaseModel):
    document_name: str
    chunk_text: str
    chunk_id: str
    page_number: Optional[int] = None
    relevance_score: float
    retrieval_method: str  # "semantic", "bm25", "hybrid"


class EvaluationResult(BaseModel):
    faithfulness_score: float = Field(ge=0, le=1)
    relevance_score: float = Field(ge=0, le=1)
    faithfulness_reasoning: str
    relevance_reasoning: str


class PipelineStep(BaseModel):
    name: str
    duration_ms: float
    details: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    evaluation: Optional[EvaluationResult] = None
    pipeline_steps: List[PipelineStep]
    conversation_id: str
    confidence: float
    is_fallback: bool = False


class DocumentInfo(BaseModel):
    id: str
    name: str
    size_bytes: int
    num_chunks: int
    uploaded_at: datetime


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int


class UploadResponse(BaseModel):
    document_id: str
    name: str
    num_chunks: int
    message: str


class ChunkData(BaseModel):
    chunk_id: str
    text: str
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    total_documents: int
    total_chunks: int
    version: str = "1.0.0"
