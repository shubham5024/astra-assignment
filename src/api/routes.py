"""FastAPI Route Handlers for Query, Ingestion, and Health endpoints."""

from fastapi import APIRouter, HTTPException, Depends, status
import time
import logging

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    CitationResponse,
    StepResponse,
    IngestResponse,
    HealthResponse,
    DocumentItem,
)
from src.ingestion.pipeline import IngestionPipeline
from src.agent.router import QueryRouter
from src.rag.pipeline import SimpleRAGPipeline, RAGResponse
from src.agent.orchestrator import AgentOrchestrator, AgentResponse
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


def get_services(request):
    """Retrieve initialized services from FastAPI application state."""
    return request.app.state.services


@router.post("/query", response_model=QueryResponse, summary="Query the Intelligent Knowledge Assistant")
async def answer_question(payload: QueryRequest):
    """
    Main endpoint for asking technical questions.
    Dynamically routes between Simple RAG and Agentic Multi-Step workflows based on query intent.
    """
    start_time = time.time()
    from src.api.app import get_app_services

    services = get_app_services()
    if not services["vector_store"].chunks:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is empty. Please run ingestion first via POST /api/v1/ingest or scripts/ingest.py",
        )

    router_inst: QueryRouter = services["router"]
    rag_pipe: SimpleRAGPipeline = services["simple_rag"]
    agent_orch: AgentOrchestrator = services["agent_orchestrator"]

    # 1. Routing Decision
    if payload.workflow_override == "simple_rag":
        target_workflow = "SIMPLE_RAG"
        routing_reason = "Manual override to Simple RAG requested by user."
    elif payload.workflow_override == "agentic_workflow":
        target_workflow = "AGENTIC_WORKFLOW"
        routing_reason = "Manual override to Agentic Workflow requested by user."
    else:
        decision = router_inst.route(payload.query)
        target_workflow = decision.workflow
        routing_reason = decision.reasoning

    logger.info(f"Query: '{payload.query}' -> Selected Workflow: {target_workflow}")

    # 2. Execution Dispatch
    if target_workflow == "DIRECT_REFUSAL":
        elapsed = round(time.time() - start_time, 4)
        return QueryResponse(
            query=payload.query,
            answer="I do not have sufficient information in the provided knowledge corpus to answer this question. The corpus is strictly focused on RAG Architecture Patterns, Vector Databases, and Agentic AI Frameworks.",
            workflow="DIRECT_REFUSAL",
            routing_reasoning=routing_reason,
            citations=[],
            steps=None,
            is_refusal=True,
            latency_seconds=elapsed,
        )

    elif target_workflow == "AGENTIC_WORKFLOW":
        agent_res: AgentResponse = agent_orch.run(payload.query)
        step_dtos = [
            StepResponse(
                step_number=s.step_number,
                thought=s.thought,
                tool_name=s.tool_name,
                tool_input=s.tool_input,
                observation=s.observation,
                latency_seconds=s.latency_seconds,
            )
            for s in agent_res.steps
        ]
        citation_dtos = [
            CitationResponse(
                document_name=c.document_name,
                page_number=c.page_number,
                raw_tag=c.raw_tag,
            )
            for c in agent_res.citations
        ]
        elapsed = round(time.time() - start_time, 4)
        return QueryResponse(
            query=payload.query,
            answer=agent_res.answer,
            workflow="AGENTIC_WORKFLOW",
            routing_reasoning=routing_reason,
            citations=citation_dtos,
            steps=step_dtos,
            is_refusal=agent_res.is_refusal,
            latency_seconds=elapsed,
        )

    else:  # SIMPLE_RAG
        rag_res: RAGResponse = rag_pipe.answer_query(
            query=payload.query,
            doc_filter=payload.doc_filter,
            top_k=payload.top_k,
        )
        citation_dtos = [
            CitationResponse(
                document_name=c.document_name,
                page_number=c.page_number,
                raw_tag=c.raw_tag,
            )
            for c in rag_res.citations
        ]
        elapsed = round(time.time() - start_time, 4)
        return QueryResponse(
            query=payload.query,
            answer=rag_res.answer,
            workflow="SIMPLE_RAG",
            routing_reasoning=routing_reason,
            citations=citation_dtos,
            steps=None,
            is_refusal=rag_res.is_refusal,
            latency_seconds=elapsed,
        )


@router.post("/ingest", response_model=IngestResponse, summary="Trigger Document Ingestion Pipeline")
async def trigger_ingestion():
    """Trigger PDF parsing, chunking, and dual-index creation."""
    try:
        from src.api.app import get_app_services, reload_app_services

        pipeline = IngestionPipeline()
        summary = pipeline.run()
        reload_app_services()

        return IngestResponse(
            success=True,
            message=f"Successfully indexed {summary.total_documents} documents ({summary.total_chunks} chunks).",
            summary=summary,
        )
    except Exception as e:
        logger.exception("Ingestion failed via API trigger")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse, summary="Service Health and Index Status")
async def health_check():
    """Return runtime service and index status."""
    from src.api.app import get_app_services

    services = get_app_services()
    vector_store = services["vector_store"]
    unique_docs = sorted(list({c.document_name for c in vector_store.chunks}))

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        index_loaded=vector_store.count() > 0,
        total_chunks=vector_store.count(),
        documents=unique_docs,
    )


@router.get("/documents", summary="List Ingested Documents and Metadata")
async def list_documents():
    """List details of all documents present in the knowledge index."""
    from src.api.app import get_app_services

    services = get_app_services()
    vector_store = services["vector_store"]

    doc_stats = {}
    for c in vector_store.chunks:
        if c.document_name not in doc_stats:
            doc_stats[c.document_name] = {"chunks": 0, "pages": set()}
        doc_stats[c.document_name]["chunks"] += 1
        doc_stats[c.document_name]["pages"].add(c.page_number)

    return {
        "total_documents": len(doc_stats),
        "documents": [
            {
                "document_name": name,
                "chunks_count": data["chunks"],
                "pages_count": len(data["pages"]),
            }
            for name, data in doc_stats.items()
        ],
    }
