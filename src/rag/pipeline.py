"""Simple Grounded RAG Pipeline."""

from typing import List, Dict, Any, Optional
import time
import logging
from pydantic import BaseModel, Field

from src.search.hybrid import HybridSearchEngine, ScoredChunk
from src.rag.generator import LLMGenerator, GenerationResult, Citation
from src.config import settings

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Unified response object for RAG queries."""
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Grounded answer text")
    citations: List[Citation] = Field(default_factory=list, description="Extracted source citations")
    workflow: str = Field(default="SIMPLE_RAG", description="Workflow path executed")
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list, description="Diagnostic chunk metadata")
    is_refusal: bool = Field(default=False, description="Whether answer is a refusal due to missing information")
    latency_seconds: float = Field(..., description="End-to-end execution latency in seconds")


class SimpleRAGPipeline:
    """Standard retrieve-then-generate pipeline with grounded citations and refusal guardrails."""

    def __init__(
        self,
        hybrid_searcher: HybridSearchEngine,
        generator: Optional[LLMGenerator] = None,
        top_k: int = None,
    ):
        self.searcher = hybrid_searcher
        self.generator = generator or LLMGenerator()
        self.top_k = top_k or settings.TOP_K_RETRIEVAL

    def answer_query(
        self,
        query: str,
        doc_filter: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute RAG flow: Retrieve top chunks -> Apply threshold -> Generate grounded answer.
        """
        start_time = time.time()
        k = top_k or self.top_k

        # 1. Retrieve candidates via Hybrid Search
        scored_chunks = self.searcher.search(
            query=query,
            top_k=k,
            doc_filter=doc_filter,
        )

        # 2. Check for empty or low-confidence results
        if not scored_chunks:
            logger.info(f"SimpleRAG: No chunks found for query '{query}'. Returning refusal.")
            elapsed = time.time() - start_time
            return RAGResponse(
                query=query,
                answer="I do not have sufficient information in the provided knowledge corpus to answer this question.",
                citations=[],
                workflow="SIMPLE_RAG",
                retrieved_chunks=[],
                is_refusal=True,
                latency_seconds=round(elapsed, 4),
            )

        # 3. Generate grounded answer
        gen_result: GenerationResult = self.generator.generate(
            question=query,
            retrieved_chunks=scored_chunks,
        )

        elapsed = time.time() - start_time
        return RAGResponse(
            query=query,
            answer=gen_result.answer,
            citations=gen_result.citations,
            workflow="SIMPLE_RAG",
            retrieved_chunks=[sc.to_dict() for sc in scored_chunks],
            is_refusal=gen_result.is_refusal,
            latency_seconds=round(elapsed, 4),
        )
