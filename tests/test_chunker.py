  poetry run python scripts/evaluate.py"""Unit tests for semantic recursive chunker."""

import pytest
from src.ingestion.chunker import SemanticRecursiveChunker


def test_chunker_basic_split():
    chunker = SemanticRecursiveChunker(chunk_size_tokens=50, chunk_overlap_tokens=10)
    sample_text = (
        "Retrieval-Augmented Generation (RAG) combines external knowledge retrieval with LLMs. "
        "Chunking strategies are critical for high-quality retrieval. "
        "Dense vectors capture semantics while sparse BM25 matches exact keywords."
    )
    chunks = chunker.chunk_page(sample_text, document_name="test_doc.pdf", page_number=1)
    
    assert len(chunks) >= 1
    assert chunks[0].document_name == "test_doc.pdf"
    assert chunks[0].page_number == 1
    assert "test_doc_pdf_p1_c000" in chunks[0].chunk_id
    assert chunks[0].token_count > 0


def test_chunker_empty_text():
    chunker = SemanticRecursiveChunker()
    chunks = chunker.chunk_page("", document_name="test.pdf", page_number=1)
    assert chunks == []


def test_chunker_large_text_overlap():
    chunker = SemanticRecursiveChunker(chunk_size_tokens=30, chunk_overlap_tokens=10)
    long_text = "Paragraph one with technical content. \n\n" * 10
    chunks = chunker.chunk_page(long_text, document_name="long_doc.pdf", page_number=2)
    
    assert len(chunks) > 1
    for c in chunks:
        assert c.page_number == 2
        assert len(c.text) > 0
