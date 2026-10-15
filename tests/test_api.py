"""End-to-End API integration tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["api_prefix"] == "/api/v1"


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["index_loaded"] is True
    assert data["total_chunks"] > 0
    assert len(data["documents"]) >= 3


def test_documents_endpoint(client):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] >= 3
    assert len(data["documents"]) >= 3


def test_query_simple_rag(client):
    payload = {
        "query": "What are the primary index types supported by FAISS?",
        "workflow_override": "simple_rag",
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["workflow"] == "SIMPLE_RAG"
    assert len(data["answer"]) > 0
    assert len(data["citations"]) >= 1


def test_query_agentic_workflow(client):
    payload = {
        "query": "Compare Pinecone and Chroma across latency, cost model, and persistence.",
        "workflow_override": "agentic_workflow",
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["workflow"] == "AGENTIC_WORKFLOW"
    assert data["steps"] is not None
    assert len(data["steps"]) >= 2
    assert len(data["citations"]) >= 1


def test_query_out_of_corpus_refusal(client):
    payload = {
        "query": "What is the best recipe for baking Italian chocolate cake?",
        "workflow_override": "auto",
    }
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["workflow"] == "DIRECT_REFUSAL"
    assert data["is_refusal"] is True
    assert "not have sufficient information" in data["answer"].lower()
