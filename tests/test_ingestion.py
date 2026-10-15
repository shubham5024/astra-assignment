"""Unit and integration tests for document parsing and ingestion."""

import pytest
from pathlib import Path
from src.ingestion.parser import PDFParser
from src.ingestion.pipeline import IngestionPipeline
from src.config import settings


def test_pdf_parser_on_corpus():
    parser = PDFParser()
    doc_path = settings.resolved_docs_dir / "rag_architecture_patterns.pdf"
    assert doc_path.exists(), f"Expected document at {doc_path}"

    parsed = parser.parse_file(doc_path)
    assert parsed["metadata"].total_pages >= 1
    assert len(parsed["pages"]) > 0
    assert parsed["pages"][0]["page_number"] == 1
    assert "RAG" in parsed["pages"][0]["text"]


def test_ingestion_pipeline_run(tmp_path):
    pipeline = IngestionPipeline()
    summary = pipeline.run(documents_dir=settings.resolved_docs_dir, storage_dir=tmp_path)

    assert summary.total_documents >= 3
    assert summary.total_chunks > 0
    assert (tmp_path / "dense_embeddings.npy").exists()
    assert (tmp_path / "dense_chunks.json").exists()
    assert (tmp_path / "bm25_index.json").exists()
