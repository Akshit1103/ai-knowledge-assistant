"""
Main RAG pipeline orchestrator — wires every stage together.
"""

import time
from typing import Dict, Any, Optional, List

from config import settings
from store.vector_store import VectorStore
from store.bm25_store import BM25Store
from rag.query_expansion import QueryExpander
from rag.retrieval import HybridRetriever
from rag.reranker import Reranker
from rag.generator import Generator
from rag.guardrails import Guardrails
from rag.evaluation import Evaluator
from rag.memory import ConversationMemory
from models.schemas import QueryResponse, Source, EvaluationResult, PipelineStep


class RAGPipeline:
    """
    Production-grade RAG pipeline:
    Query → Guardrails → Expansion → Hybrid Retrieval → Re-rank → Generate → Evaluate
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.expander = QueryExpander()
        self.retriever = HybridRetriever(vector_store, bm25_store)
        self.reranker = Reranker()
        self.generator = Generator()
        self.guardrails = Guardrails()
        self.evaluator = Evaluator()
        self.memory = ConversationMemory()

    def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        include_evaluation: bool = True,
    ) -> QueryResponse:
        """Execute the full RAG pipeline and return a structured response."""
        pipeline_steps: List[PipelineStep] = []

        # ----------------------------------------------------------
        # 1. Input guardrails
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        input_check = self.guardrails.check_input(query)
        pipeline_steps.append(
            PipelineStep(
                name="Input Guardrails",
                duration_ms=_elapsed_ms(t0),
                details={"allowed": input_check["allowed"], "reason": input_check.get("reason")},
            )
        )
        if not input_check["allowed"]:
            session_id = self.memory.get_or_create_session(conversation_id)
            return QueryResponse(
                answer=input_check["reason"],
                sources=[],
                pipeline_steps=pipeline_steps,
                conversation_id=session_id,
                confidence=0.0,
                is_fallback=True,
            )

        # ----------------------------------------------------------
        # 2. Conversation memory
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        session_id = self.memory.get_or_create_session(conversation_id)
        history = self.memory.get_history(session_id)
        pipeline_steps.append(
            PipelineStep(
                name="Conversation Memory",
                duration_ms=_elapsed_ms(t0),
                details={"session_id": session_id, "history_length": len(history)},
            )
        )

        # ----------------------------------------------------------
        # 3. Query expansion
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        expanded_queries = self.expander.expand(query)
        pipeline_steps.append(
            PipelineStep(
                name="Query Expansion",
                duration_ms=_elapsed_ms(t0),
                details={"original": query, "expanded": expanded_queries},
            )
        )

        # ----------------------------------------------------------
        # 4. Hybrid retrieval
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        retrieved = self.retriever.retrieve(expanded_queries)
        pipeline_steps.append(
            PipelineStep(
                name="Hybrid Retrieval",
                duration_ms=_elapsed_ms(t0),
                details={"total_candidates": len(retrieved)},
            )
        )

        # ----------------------------------------------------------
        # 5. Re-ranking
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, retrieved)
        pipeline_steps.append(
            PipelineStep(
                name="Re-ranking",
                duration_ms=_elapsed_ms(t0),
                details={"kept": len(reranked)},
            )
        )

        # ----------------------------------------------------------
        # 6. Generation
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        gen_result = self.generator.generate(query, reranked, history)
        pipeline_steps.append(
            PipelineStep(
                name="LLM Generation",
                duration_ms=_elapsed_ms(t0),
                details={"confidence": gen_result["confidence"]},
            )
        )

        # ----------------------------------------------------------
        # 7. Output guardrails
        # ----------------------------------------------------------
        t0 = time.perf_counter()
        output_check = self.guardrails.check_output(
            gen_result["answer"], gen_result["confidence"], reranked
        )
        pipeline_steps.append(
            PipelineStep(
                name="Output Guardrails",
                duration_ms=_elapsed_ms(t0),
                details={"is_fallback": output_check["is_fallback"]},
            )
        )

        # ----------------------------------------------------------
        # 8. Evaluation (optional)
        # ----------------------------------------------------------
        evaluation = None
        if include_evaluation and reranked:
            t0 = time.perf_counter()
            eval_result = self.evaluator.evaluate(query, output_check["answer"], reranked)
            evaluation = EvaluationResult(**eval_result)
            pipeline_steps.append(
                PipelineStep(
                    name="Evaluation",
                    duration_ms=_elapsed_ms(t0),
                    details={
                        "faithfulness": eval_result["faithfulness_score"],
                        "relevance": eval_result["relevance_score"],
                    },
                )
            )

        # ----------------------------------------------------------
        # 9. Save to conversation memory
        # ----------------------------------------------------------
        self.memory.add_message(session_id, "user", query)
        self.memory.add_message(session_id, "assistant", output_check["answer"])

        # ----------------------------------------------------------
        # Build structured sources
        # ----------------------------------------------------------
        sources = []
        for chunk in reranked:
            meta = chunk.get("metadata", {})
            sources.append(
                Source(
                    document_name=meta.get("document_name", "Unknown"),
                    chunk_text=chunk["text"],
                    chunk_id=chunk["chunk_id"],
                    page_number=meta.get("page_number"),
                    relevance_score=chunk.get("relevance_score", 0.0),
                    retrieval_method=chunk.get("retrieval_method", "hybrid"),
                )
            )

        return QueryResponse(
            answer=output_check["answer"],
            sources=sources,
            evaluation=evaluation,
            pipeline_steps=pipeline_steps,
            conversation_id=session_id,
            confidence=output_check["confidence"],
            is_fallback=output_check["is_fallback"],
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
