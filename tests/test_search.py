"""Unit tests for Dense, Sparse BM25, and Hybrid RRF search engines."""

import pytest
import numpy as np
from src.ingestion.models import DocumentChunk
from src.vector_store.store import DenseVectorStore
from src.search.bm25 import BM25SearchEngine
from src.search.hybrid import HybridSearchEngine
from src.embeddings.base import DeterministicFallbackEmbedder


@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="doc1_p1_c001",
            document_name="rag_architecture_patterns.pdf",
            page_number=1,
            chunk_index=0,
            text="Chunking strategies include fixed-size and semantic chunking with overlap.",
            token_count=12,
        ),
        DocumentChunk(
            chunk_id="doc2_p1_c001",
            document_name="vector_database_comparison.pdf",
            page_number=1,
            chunk_index=0,
            text="Chroma and Pinecone are popular vector databases for fast similarity search.",
            token_count=12,
        ),
        DocumentChunk(
            chunk_id="doc3_p1_c001",
            document_name="agentic_ai_frameworks.pdf",
            page_number=1,
            chunk_index=0,
            text="LangGraph provides cyclic state machine graphs for multi-agent workflows.",
            token_count=11,
        ),
    ]


def test_bm25_search(sample_chunks):
    bm25 = BM25SearchEngine()
    bm25.build_index(sample_chunks)

    results = bm25.search("LangGraph state machine", top_k=2)
    assert len(results) >= 1
    assert results[0][0].document_name == "agentic_ai_frameworks.pdf"


def test_dense_store_search(sample_chunks):
    embedder = DeterministicFallbackEmbedder(dimension=64)
    texts = [c.text for c in sample_chunks]
    embeddings = embedder.embed_texts(texts)

    store = DenseVectorStore()
    store.add_chunks(sample_chunks, embeddings)

    q_vec = embedder.embed_query("vector databases similarity search")
    results = store.search(q_vec, top_k=2)

    assert len(results) >= 1
    assert results[0][0].document_name == "vector_database_comparison.pdf"


def test_hybrid_search(sample_chunks):
    embedder = DeterministicFallbackEmbedder(dimension=64)
    texts = [c.text for c in sample_chunks]
    embeddings = embedder.embed_texts(texts)

    store = DenseVectorStore()
    store.add_chunks(sample_chunks, embeddings)

    bm25 = BM25SearchEngine()
    bm25.build_index(sample_chunks)

    hybrid = HybridSearchEngine(
        vector_store=store,
        bm25_engine=bm25,
        embedder=embedder,
        similarity_threshold=0.0,
    )

    scored = hybrid.search("fixed-size chunking overlap", top_k=2)
    assert len(scored) >= 1
    assert scored[0].chunk.document_name == "rag_architecture_patterns.pdf"
    assert scored[0].rrf_score > 0
