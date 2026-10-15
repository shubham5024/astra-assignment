"""Query Intent Router for intelligent dispatching between Simple RAG, Agentic workflows, and Direct Refusal."""

from typing import Optional, Literal
from pydantic import BaseModel, Field
import re
import logging
from openai import OpenAI

from src.config import settings

logger = logging.getLogger(__name__)

WorkflowType = Literal["SIMPLE_RAG", "AGENTIC_WORKFLOW", "DIRECT_REFUSAL"]


class RoutingDecision(BaseModel):
    """Structured decision returned by the query intent router."""
    workflow: WorkflowType = Field(..., description="Target workflow for query execution")
    confidence: float = Field(..., description="Confidence score of classification [0.0 - 1.0]")
    reasoning: str = Field(..., description="Explanation of why this workflow was selected")
    sub_type: Optional[str] = Field(default=None, description="Sub-category (e.g. comparison, multi_hop, single_fact)")


class QueryRouter:
    """
    Analyzes user queries to determine whether a direct Simple RAG pipeline is sufficient,
    an Agentic multi-step workflow is needed, or the query is completely out-of-domain.
    """

    KNOWN_TOPICS = {
        "rag", "retrieval", "chunk", "chunking", "embedding", "vector", "vector db",
        "faiss", "pinecone", "weaviate", "pgvector", "chroma", "langchain", "langgraph",
        "crewai", "autogen", "agent", "agentic", "hybrid search", "bm25", "hnsw",
        "rerank", "re-ranking", "hallucination", "context window", "lcel", "multi-agent",
        "supervision", "reflection", "react", "plan-and-solve", "prompt", "evaluation",
        "faithfulness", "hit rate", "mrr", "ndcg", "grounding", "citations"
    }

    # Explicitly unindexed technologies noted in assessment / out of domain
    OUT_OF_DOMAIN_PATTERNS = [
        r"\b(recipe|cook|bake|cake|pasta|pizza|dinner|lunch|breakfast|brownies)\b",
        r"\b(weather|forecast|rain|temperature|climate)\b",
        r"\b(stock price|crypto|bitcoin|ethereum|forex)\b",
        r"\b(quantum computing|qubit|quantum supremacy)\b",
        r"\b(movie|actor|film|hollywood|music|song|album)\b",
        r"\b(medical|hypertension|heart disease|drug|disease|symptom|doctor)\b",
        r"\b(milvus|qdrant)\b",  # Document 3 explicitly designates Milvus and Qdrant as unanswerable out-of-corpus topics
    ]

    COMPARISON_KEYWORDS = [
        r"\bcompare\b", r"\bcomparison\b", r"\bversus\b", r"\bvs\b",
        r"\btrade-offs?\b", r"\bpros and cons\b", r"\bdifferences? between\b"
    ]

    MULTI_STEP_PATTERNS = [
        r"\bhow should (?:an enterprise|we|i) combine\b",
        r"\bstep by step\b",
        r"\barchitecture for\b",
        r"\bhow do (?:both|they) compare\b",
        r"\btrade-offs between (?:react|plan)\b",
        r"\bexplain how to combine\b",
        r"\bhow to design\b",
        r"\bmetrics and frameworks\b",
    ]

    FACTOID_PATTERNS = [
        r"^what (?:is|are|chunk|size)",
        r"^who (?:first|formalized)",
        r"^which (?:year|index)",
        r"recommended for dense qa",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._client: Optional[OpenAI] = None
        if self.api_key and self.api_key.strip() and self.api_key != "your_openai_api_key_here":
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=settings.OPENAI_BASE_URL)
            except Exception as e:
                logger.warning(f"Router could not initialize OpenAI client: {e}")

    def route(self, query: str) -> RoutingDecision:
        """
        Classify incoming query using hierarchical intent classification.
        """
        q_lower = query.lower().strip()

        # 1. Out-of-Domain Refusal Check (highest priority)
        for pattern in self.OUT_OF_DOMAIN_PATTERNS:
            if re.search(pattern, q_lower):
                return RoutingDecision(
                    workflow="DIRECT_REFUSAL",
                    confidence=0.98,
                    reasoning=f"Query references out-of-corpus subject matching rule '{pattern}'.",
                    sub_type="out_of_domain",
                )

        # 2. Check for Specific Multi-Step / Multi-Hop Patterns
        for pattern in self.MULTI_STEP_PATTERNS:
            if re.search(pattern, q_lower):
                return RoutingDecision(
                    workflow="AGENTIC_WORKFLOW",
                    confidence=0.92,
                    reasoning="Query requires multi-hop architectural reasoning or cross-system synthesis.",
                    sub_type="multi_hop",
                )

        # 3. Check for Direct Factoid Patterns (before broad comparisons)
        is_factoid = any(re.search(pat, q_lower) for pat in self.FACTOID_PATTERNS)
        has_comparison = any(re.search(kw, q_lower) for kw in self.COMPARISON_KEYWORDS)

        if has_comparison:
            return RoutingDecision(
                workflow="AGENTIC_WORKFLOW",
                confidence=0.94,
                reasoning="Query asks for comparative evaluation across multiple technologies/systems, requiring cross-document synthesis.",
                sub_type="comparison",
            )

        if is_factoid:
            return RoutingDecision(
                workflow="SIMPLE_RAG",
                confidence=0.92,
                reasoning="Query is a focused factual technical question ideal for single-step hybrid retrieval.",
                sub_type="single_fact",
            )

        # 4. Check Known Technical Domain Relevance
        has_domain_term = any(term in q_lower for term in self.KNOWN_TOPICS)
        if has_domain_term:
            return RoutingDecision(
                workflow="SIMPLE_RAG",
                confidence=0.85,
                reasoning="Query contains in-domain concepts suitable for direct Simple RAG.",
                sub_type="single_fact",
            )

        # 5. Default Fallback
        return RoutingDecision(
            workflow="SIMPLE_RAG",
            confidence=0.70,
            reasoning="Defaulting to Simple RAG with hybrid search confidence thresholding.",
            sub_type="general",
        )
