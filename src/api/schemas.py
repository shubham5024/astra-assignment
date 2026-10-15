"""Pydantic API request and response schemas."""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from src.ingestion.models import IngestionSummary


class QueryRequest(BaseModel):
    """Incoming user query payload."""
    query: str = Field(..., min_length=1, max_length=2000, description="The technical question to answer")
    workflow_override: Optional[Literal["auto", "simple_rag", "agentic_workflow"]] = Field(
        default="auto",
        description="Override automatic intent routing ('auto', 'simple_rag', 'agentic_workflow')",
    )
    doc_filter: Optional[str] = Field(
        default=None,
        description="Optional filter for specific document filename (e.g. 'rag_architecture_patterns.pdf')",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of retrieval chunks to retrieve for generation",
    )


class CitationResponse(BaseModel):
    """Source attribution citation."""
    document_name: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="1-indexed page number in the source document")
    raw_tag: str = Field(..., description="Formatted citation tag")


class StepResponse(BaseModel):
    """A reasoning/tool execution step from the agentic workflow."""
    step_number: int = Field(..., description="1-indexed step number")
    thought: str = Field(..., description="Internal reasoning thought")
    tool_name: Optional[str] = Field(default=None, description="Tool invoked")
    tool_input: Optional[Dict[str, Any]] = Field(default=None, description="Arguments passed to tool")
    observation: Optional[str] = Field(default=None, description="Result observed from tool")
    latency_seconds: float = Field(..., description="Step duration in seconds")


class QueryResponse(BaseModel):
    """Comprehensive structured response for a query."""
    query: str = Field(..., description="Original question asked")
    answer: str = Field(..., description="Synthesized grounded answer")
    workflow: str = Field(..., description="Workflow executed: SIMPLE_RAG, AGENTIC_WORKFLOW, or DIRECT_REFUSAL")
    routing_reasoning: Optional[str] = Field(default=None, description="Explanation for workflow selection")
    citations: List[CitationResponse] = Field(default_factory=list, description="Grounding source citations")
    steps: Optional[List[StepResponse]] = Field(default=None, description="Trace of agent steps (if agentic)")
    is_refusal: bool = Field(default=False, description="Whether answer is a refusal due to missing knowledge")
    latency_seconds: float = Field(..., description="Total end-to-end processing time in seconds")


class IngestResponse(BaseModel):
    """Response returned upon triggering knowledge ingestion."""
    success: bool = Field(..., description="Status of ingestion execution")
    message: str = Field(..., description="Human-readable status message")
    summary: Optional[IngestionSummary] = Field(default=None, description="Detailed ingestion metrics")


class DocumentItem(BaseModel):
    """Summary record of an ingested document."""
    document_name: str
    chunks_count: int
    pages_count: int


class HealthResponse(BaseModel):
    """System health and index status response."""
    status: str = Field(default="healthy")
    version: str = Field(default="0.1.0")
    index_loaded: bool = Field(..., description="Whether vector & BM25 indexes are loaded into memory")
    total_chunks: int = Field(..., description="Total chunks currently searchable")
    documents: List[str] = Field(default_factory=list, description="List of indexed document filenames")
