"""Unit tests for Query Router, Agent Tools, and ReAct Orchestrator."""

import pytest
from src.agent.router import QueryRouter
from src.agent.tools import AgentToolRegistry
from src.agent.orchestrator import AgentOrchestrator
from src.search.hybrid import HybridSearchEngine
from src.vector_store.store import DenseVectorStore
from src.search.bm25 import BM25SearchEngine
from src.embeddings.base import DeterministicFallbackEmbedder
from src.ingestion.models import DocumentChunk


@pytest.fixture
def mock_search_engine():
    chunks = [
        DocumentChunk(
            chunk_id="chunk_1",
            document_name="vector_database_comparison.pdf",
            page_number=2,
            chunk_index=0,
            text="FAISS supports Flat, IVF, and HNSW index types with GPU acceleration.",
            token_count=12,
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_name="agentic_ai_frameworks.pdf",
            page_number=3,
            chunk_index=0,
            text="CrewAI orchestrates role-playing multi-agent systems with hierarchical delegation.",
            token_count=11,
        ),
    ]
    embedder = DeterministicFallbackEmbedder(dimension=64)
    embeddings = embedder.embed_texts([c.text for c in chunks])

    store = DenseVectorStore()
    store.add_chunks(chunks, embeddings)

    bm25 = BM25SearchEngine()
    bm25.build_index(chunks)

    return HybridSearchEngine(vector_store=store, bm25_engine=bm25, embedder=embedder, similarity_threshold=0.0)


def test_query_router_routing():
    router = QueryRouter()

    # Factoid -> SIMPLE_RAG
    dec1 = router.route("What are the index types in FAISS?")
    assert dec1.workflow == "SIMPLE_RAG"

    # Comparison -> AGENTIC_WORKFLOW
    dec2 = router.route("Compare FAISS and Pinecone for production latency and cost.")
    assert dec2.workflow == "AGENTIC_WORKFLOW"

    # Out of Domain -> DIRECT_REFUSAL
    dec3 = router.route("How to bake chocolate cupcakes?")
    assert dec3.workflow == "DIRECT_REFUSAL"

    # Explicitly unindexed tech -> DIRECT_REFUSAL
    dec4 = router.route("Compare Milvus with Qdrant for billions of items.")
    assert dec4.workflow == "DIRECT_REFUSAL"


def test_agent_tool_registry(mock_search_engine):
    registry = AgentToolRegistry(search_engine=mock_search_engine)
    tools = registry.get_tool_definitions()
    assert len(tools) == 3

    res = registry.execute_tool("search_knowledge_base", {"query": "FAISS GPU"})
    assert res.success is True
    assert len(res.citations) >= 1
    assert "vector_database_comparison.pdf" in res.output


def test_agent_orchestrator_deterministic_run(mock_search_engine):
    registry = AgentToolRegistry(search_engine=mock_search_engine)
    orchestrator = AgentOrchestrator(tool_registry=registry)

    response = orchestrator.run("Compare FAISS and CrewAI capabilities.")
    assert response.workflow == "AGENTIC_WORKFLOW"
    assert len(response.steps) >= 2
    assert len(response.citations) >= 1
