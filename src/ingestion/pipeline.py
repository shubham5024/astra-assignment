"""End-to-end Document Ingestion and Dual-Index Building Pipeline."""

from pathlib import Path
from typing import List, Dict, Optional
import logging
import time

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import SemanticRecursiveChunker
from src.ingestion.models import DocumentChunk, IngestionSummary
from src.embeddings.base import BaseEmbedder
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.vector_store.store import DenseVectorStore
from src.search.bm25 import BM25SearchEngine
from src.config import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Coordinates parsing, chunking, embedding generation, and dual-index persistence."""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        chunk_size_tokens: int = None,
        chunk_overlap_tokens: int = None,
    ):
        self.embedder = embedder or OpenAIEmbedder()
        self.parser = PDFParser(clean_whitespace=True)
        self.chunker = SemanticRecursiveChunker(
            chunk_size_tokens=chunk_size_tokens or settings.CHUNK_SIZE_TOKENS,
            chunk_overlap_tokens=chunk_overlap_tokens or settings.CHUNK_OVERLAP_TOKENS,
        )

    def run(
        self,
        documents_dir: Optional[Path] = None,
        storage_dir: Optional[Path] = None,
    ) -> IngestionSummary:
        """
        Execute ingestion over all PDF documents in documents_dir and save indexes to storage_dir.
        """
        doc_dir = Path(documents_dir) if documents_dir else settings.resolved_docs_dir
        store_dir = Path(storage_dir) if storage_dir else settings.resolved_storage_dir
        store_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting ingestion pipeline from {doc_dir} -> {store_dir}")
        pdf_files = sorted(list(doc_dir.glob("*.pdf")))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF documents found in directory: {doc_dir}")

        all_chunks: List[DocumentChunk] = []
        document_breakdown: Dict[str, Dict[str, int]] = {}
        total_pages = 0

        start_time = time.time()
        for pdf_path in pdf_files:
            logger.info(f"Processing document: {pdf_path.name}")
            parsed = self.parser.parse_file(pdf_path)
            meta = parsed["metadata"]
            pages = parsed["pages"]
            total_pages += meta.total_pages

            doc_chunks: List[DocumentChunk] = []
            chunk_counter = 0
            for page in pages:
                page_chunks = self.chunker.chunk_page(
                    text=page["text"],
                    document_name=pdf_path.name,
                    page_number=page["page_number"],
                    start_chunk_index=chunk_counter,
                )
                chunk_counter += len(page_chunks)
                doc_chunks.extend(page_chunks)

            all_chunks.extend(doc_chunks)
            document_breakdown[pdf_path.name] = {
                "pages": meta.total_pages,
                "text_pages": len(pages),
                "chunks": len(doc_chunks),
            }
            logger.info(f"Extracted {len(doc_chunks)} chunks from {pdf_path.name} ({meta.total_pages} pages).")

        # 1. Generate Embeddings & Save Dense Vector Store
        logger.info(f"Generating embeddings for {len(all_chunks)} total chunks...")
        texts = [chunk.text for chunk in all_chunks]
        embeddings = self.embedder.embed_texts(texts)

        vector_store = DenseVectorStore(storage_dir=store_dir)
        vector_store.add_chunks(all_chunks, embeddings)
        vector_store.save(store_dir)

        # 2. Build & Save Sparse BM25 Index
        logger.info("Building BM25 inverted index...")
        bm25_engine = BM25SearchEngine(storage_dir=store_dir)
        bm25_engine.build_index(all_chunks)
        bm25_engine.save(store_dir)

        elapsed = time.time() - start_time
        logger.info(f"Ingestion completed in {elapsed:.2f}s: {len(pdf_files)} docs, {total_pages} pages, {len(all_chunks)} chunks.")

        return IngestionSummary(
            total_documents=len(pdf_files),
            total_pages=total_pages,
            total_chunks=len(all_chunks),
            document_breakdown=document_breakdown,
            storage_path=str(store_dir),
        )
