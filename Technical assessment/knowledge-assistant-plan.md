# Intelligent Knowledge Assistant — Implementation Plan

## Top-Level Overview
Build a production-grade, human-authored Intelligent Knowledge Assistant in Python using Poetry for dependency management. The system processes an unstructured technical PDF corpus (`rag_architecture_patterns.pdf`, `vector_database_comparison.pdf`, `agentic_ai_frameworks.pdf`) through an offline ingestion pipeline with semantic-aware recursive chunking and dual indexing (Dense HNSW + Sparse BM25). At runtime, an intelligent Query Router dispatches requests to either a high-precision Simple Hybrid RAG pipeline or a Multi-Step Agentic Orchestrator with tool-calling capabilities. Responses are strictly grounded with page-level citations, accompanied by automated evaluation benchmarks, unit/e2e test suites, a FastAPI REST service, and a Streamlit interactive demo interface.

---

## Sub-Tasks

### Sub-Task 1: Project Scaffolding & Poetry Configuration
- **Intent**: Establish a clean, standard Python project structure with Poetry dependency management and environment configuration.
- **Expected Outcomes**:
  - Valid `pyproject.toml` with pinned core dependencies (`fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `pypdf`, `rank-bm25`, `chromadb`, `numpy`, `openai`, `streamlit`, `httpx`, `pytest`, `rich`).
  - `.env.example` defining configurable parameters (API keys, index paths, chunk size, fusion weights, similarity thresholds).
  - Clean modular directory layout (`src/`, `data/`, `scripts/`, `ui/`, `tests/`).
- **Todo List**:
  - [ ] Initialize `pyproject.toml` with project metadata and dependencies.
  - [ ] Create `.env.example` with documented environment variables.
  - [ ] Create `.gitignore` ignoring `.venv`, `__pycache__`, data stores, and environment secrets.
  - [ ] Set up `src/config.py` using `pydantic-settings` for robust runtime configuration.
- **Relevant Context**: Root directory, `Senior_AI_Engineer_Technical_Assessment.pdf`.
- **Status**: [ ] pending

---

### Sub-Task 2: Document Ingestion, Semantic Chunking & Metadata Pipeline
- **Intent**: Ingest unstructured PDF documents, extract text page-by-page, apply semantic recursive chunking with overlap, and enrich each chunk with granular metadata.
- **Expected Outcomes**:
  - PDF parser preserving exact 1-indexed page numbers and document titles.
  - Semantic-aware recursive chunker splitting on natural boundaries (headers, paragraphs, sentences) with configurable token size (~650 tokens) and overlap (15%).
  - Deterministic chunk ID generation and metadata extraction.
  - Decoupled ingestion CLI script (`scripts/ingest.py`) that processes documents once and avoids re-processing during query runtime.
- **Todo List**:
  - [ ] Implement PDF text extractor in `src/ingestion/parser.py`.
  - [ ] Implement recursive semantic chunker in `src/ingestion/chunker.py`.
  - [ ] Implement data models (`DocumentChunk`, `IngestionResult`) in `src/ingestion/models.py`.
  - [ ] Build end-to-end ingestion pipeline in `src/ingestion/pipeline.py` and CLI in `scripts/ingest.py`.
- **Relevant Context**: `data/documents/` containing corpus PDFs.
- **Status**: [ ] pending

---

### Sub-Task 3: Dual Indexing (Dense HNSW + Sparse BM25) & Hybrid Search Engine
- **Intent**: Implement dual indexing and Reciprocal Rank Fusion (RRF) retrieval combining semantic similarity and exact keyword precision with confidence thresholding.
- **Expected Outcomes**:
  - Dense vector store using persistent HNSW indexing and OpenAI embeddings (`text-embedding-3-small` with local fallback vector index).
  - Sparse BM25 inverted index with tokenization and corpus persistence.
  - Hybrid search retriever combining dense and sparse rankings using Reciprocal Rank Fusion:
    `RRF_Score(d) = alpha / (60 + rank_dense) + (1 - alpha) / (60 + rank_bm25)`.
  - Configurable relevance score filtering to eliminate low-confidence chunks and prevent hallucinations.
- **Todo List**:
  - [ ] Implement embedding client interface in `src/embeddings/base.py`.
  - [ ] Implement persistent HNSW vector store abstraction in `src/vector_store/store.py`.
  - [ ] Implement BM25 indexer and storage in `src/search/bm25.py`.
  - [ ] Implement hybrid retrieval engine with RRF fusion in `src/search/hybrid.py`.
- **Relevant Context**: `src/vector_store/`, `src/search/`.
- **Status**: [ ] pending

---

### Sub-Task 4: Grounded Simple RAG Pipeline & Refusal Engine
- **Intent**: Deliver high-precision direct question answering grounded in retrieved chunks with page-level citations and graceful refusal on out-of-corpus queries.
- **Expected Outcomes**:
  - Context assembler formatting retrieved chunks with explicit document and page metadata.
  - Grounded answer generator producing citations formatted as `[Source: <doc_name>, Page: <page_no>]`.
  - Strict out-of-domain / low-confidence refusal handler returning explicit notices when context is insufficient.
- **Todo List**:
  - [ ] Implement prompt templates and context formatter in `src/rag/prompts.py`.
  - [ ] Implement grounded generator with structured citation extractor in `src/rag/generator.py`.
  - [ ] Implement simple RAG orchestrator in `src/rag/pipeline.py`.
- **Relevant Context**: `src/rag/`.
- **Status**: [ ] pending

---

### Sub-Task 5: Agentic Workflow & Multi-Step Reasoning Engine
- **Intent**: Enable multi-step reasoning, cross-document comparison, query decomposition, and dynamic tool execution for complex questions.
- **Expected Outcomes**:
  - Query Router classifying incoming queries into `SIMPLE_RAG`, `AGENTIC_MULTI_HOP`, `AGENTIC_COMPARISON`, or `UNANSWERABLE_DIRECT`.
  - Tool registry providing search, multi-document comparison, and section summarization capabilities.
  - Multi-step ReAct state machine orchestrator tracking execution history, intermediate evidence, and synthesizing final answers with grounding validation.
- **Todo List**:
  - [ ] Implement query classification router in `src/agent/router.py`.
  - [ ] Implement agent tools (`search_knowledge_base`, `compare_documents`, `summarize_section`) in `src/agent/tools.py`.
  - [ ] Implement ReAct state machine orchestrator and synthesis loop in `src/agent/orchestrator.py`.
- **Relevant Context**: `src/agent/`.
- **Status**: [ ] pending

---

### Sub-Task 6: FastAPI REST API & Request/Response Contracts
- **Intent**: Expose the solution via production-ready REST endpoints with typed Pydantic contracts and error handling.
- **Expected Outcomes**:
  - Endpoints: `POST /api/v1/query`, `POST /api/v1/ingest`, `GET /api/v1/health`, `GET /api/v1/documents`.
  - Structured responses containing answer, citations, workflow mode used, latency, and step execution traces.
  - Middleware for request logging, CORS, and standardized HTTP exception handlers.
- **Todo List**:
  - [ ] Define Pydantic request/response schemas in `src/api/schemas.py`.
  - [ ] Implement API route handlers in `src/api/routes.py`.
  - [ ] Assemble FastAPI application with lifespan management in `src/api/app.py`.
- **Relevant Context**: `src/api/`.
- **Status**: [ ] pending

---

### Sub-Task 7: Streamlit Interactive UI Demo
- **Intent**: Provide a lightweight, intuitive web interface to demonstrate Simple RAG, Multi-Step Agentic workflows, and unanswerable question handling.
- **Expected Outcomes**:
  - Interactive chat interface with preset question buttons covering different workflow scenarios.
  - Workflow badge (Simple RAG vs Multi-Step Agentic) and latency display.
  - Expandable citation drawer displaying document names, page numbers, relevance scores, and text excerpts.
  - Execution trace expander showing agent thoughts and intermediate tool calls.
- **Todo List**:
  - [ ] Build Streamlit UI in `ui/app.py` connecting to the FastAPI backend or local runtime.
  - [ ] Add sample prompt selector for easy demonstration.
- **Relevant Context**: `ui/app.py`.
- **Status**: [ ] pending

---

### Sub-Task 8: Evaluation Suite & 15-Question Benchmark Dataset
- **Intent**: Establish an automated evaluation benchmark to measure retrieval precision, answer grounding, refusal accuracy, and routing correctness.
- **Expected Outcomes**:
  - Curated benchmark dataset of 15 questions categorized into:
    1. Single-Document Factual
    2. Multi-Document Comparison
    3. Ambiguous / Nuanced Queries
    4. Out-of-Corpus / Unanswerable Queries
  - Automated evaluation harness computing Hit Rate@k, Grounding Faithfulness, Refusal Accuracy, and Routing Precision.
  - CLI runner in `scripts/evaluate.py` generating an evaluation summary report.
- **Todo List**:
  - [ ] Create benchmark dataset in `src/evaluation/benchmark.json`.
  - [ ] Implement evaluation metrics and runner in `src/evaluation/evaluator.py`.
  - [ ] Create CLI evaluation script in `scripts/evaluate.py`.
- **Relevant Context**: `src/evaluation/`.
- **Status**: [ ] pending

---

### Sub-Task 9: Unit & End-to-End Test Suite
- **Intent**: Ensure comprehensive automated test coverage across all pipeline layers.
- **Expected Outcomes**:
  - Unit tests for chunker, parser, BM25 search, HNSW store, hybrid fusion, query router, and agent tools.
  - Integration/API test verifying `/api/v1/query` and health endpoints.
- **Todo List**:
  - [ ] Write unit tests for chunking and ingestion in `tests/test_chunker.py` and `tests/test_ingestion.py`.
  - [ ] Write unit tests for hybrid search in `tests/test_search.py`.
  - [ ] Write unit tests for router and agent in `tests/test_agent.py`.
  - [ ] Write end-to-end API tests in `tests/test_api.py`.
- **Relevant Context**: `tests/`.
- **Status**: [ ] pending

---

### Sub-Task 10: Containerization & Deployment Setup
- **Intent**: Provide single-command reproducible containerized deployment for both API and UI.
- **Expected Outcomes**:
  - Multi-stage `Dockerfile` leveraging Poetry for fast caching and minimal production image size.
  - `docker-compose.yml` orchestrating API service and Streamlit frontend.
- **Todo List**:
  - [ ] Create `Dockerfile` with multi-stage build.
  - [ ] Create `docker-compose.yml` defining services, health checks, ports, and volumes.
- **Relevant Context**: Root directory.
- **Status**: [ ] pending

---

### Sub-Task 11: Production Documentation & Architecture Reference
- **Intent**: Deliver comprehensive, senior-level documentation explaining design rationale, trade-offs, architecture, and reproduction steps.
- **Expected Outcomes**:
  - `README.md` covering architecture, setup/run instructions, API curl examples, evaluation findings, security practices, and future production improvements.
  - `ARCHITECTURE.md` detailing technical decisions, latency/cost/scaling trade-offs, and monitoring strategies.
- **Todo List**:
  - [ ] Write comprehensive `README.md`.
  - [ ] Write in-depth `ARCHITECTURE.md` with system design analysis.
- **Relevant Context**: Root directory.
- **Status**: [ ] pending
