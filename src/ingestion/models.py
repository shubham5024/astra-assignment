"""Data models for document ingestion, chunking, and metadata management."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an ingested document source."""
    document_name: str = Field(..., description="Filename of the source document")
    file_path: str = Field(..., description="Path to the document file")
    total_pages: int = Field(..., description="Total number of pages in the document")
    document_title: Optional[str] = Field(default=None, description="Inferred or extracted title")


class DocumentChunk(BaseModel):
    """A granular chunk of text extracted from a source document."""
    chunk_id: str = Field(..., description="Unique deterministic identifier for the chunk")
    document_name: str = Field(..., description="Filename of the parent document")
    page_number: int = Field(..., description="1-indexed page number where this chunk originates")
    chunk_index: int = Field(..., description="Sequential index of the chunk within the page/document")
    text: str = Field(..., description="Extracted chunk text content")
    token_count: int = Field(..., description="Estimated or exact token count of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")

    def to_citation_header(self) -> str:
        """Format source citation string for grounding."""
        return f"[Source: {self.document_name}, Page: {self.page_number}]"


class IngestionSummary(BaseModel):
    """Summary of the ingestion pipeline execution."""
    total_documents: int = Field(..., description="Number of documents processed")
    total_pages: int = Field(..., description="Total pages extracted across all documents")
    total_chunks: int = Field(..., description="Total chunks generated and indexed")
    document_breakdown: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Per-document breakdown of pages and chunks",
    )
    storage_path: str = Field(..., description="Directory where indexes are saved")
