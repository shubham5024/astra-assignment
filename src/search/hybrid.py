"""Hybrid Search Engine with Reciprocal Rank Fusion (RRF) and confidence scoring."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np

from src.ingestion.models import DocumentChunk
from src.embeddings.base import BaseEmbedder
from src.vector_store.store import DenseVectorStore
from src.search.bm25 import BM25SearchEngine
from src.config import settings

logger = logging.getLogger(__name__)


class ScoredChunk:
    """Wrapper holding a retrieved DocumentChunk, its fused RRF score, and retrieval diagnostics."""
    def __init__(
        self,
        chunk: DocumentChunk,
        rrf_score: float,
        dense_score: Optional[float] = None,
        sparse_score: Optional[float] = None,
        dense_rank: Optional[int] = None,
        sparse_rank: Optional[int] = None,
    ):
        self.chunk = chunk
        self.rrf_score = rrf_score
        self.dense_score = dense_score
        self.sparse_score = sparse_score
        self.dense_rank = dense_rank
        self.sparse_rank = sparse_rank

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "document_name": self.chunk.document_name,
            "page_number": self.chunk.page_number,
            "chunk_index": self.chunk.chunk_index,
            "text": self.chunk.text,
            "token_count": self.chunk.token_count,
            "rrf_score": round(self.rrf_score, 5),
            "dense_score": round(self.dense_score, 4) if self.dense_score is not None else None,
            "sparse_score": round(self.sparse_score, 4) if self.sparse_score is not None else None,
            "citation": self.chunk.to_citation_header(),
        }


class HybridSearchEngine:
    """
    Combines Dense HNSW/Cosine retrieval with Sparse BM25 retrieval
    using Reciprocal Rank Fusion (RRF) with configurable weighting.
    """

    def __init__(
        self,
        vector_store: DenseVectorStore,
        bm25_engine: BM25SearchEngine,
        embedder: BaseEmbedder,
        rrf_k: int = None,
        dense_weight: float = None,
        sparse_weight: float = None,
        similarity_threshold: float = None,
    ):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        self.embedder = embedder
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        self.dense_weight = dense_weight if dense_weight is not None else settings.DENSE_WEIGHT
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.SPARSE_WEIGHT
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.SIMILARITY_THRESHOLD
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        min_dense_similarity: Optional[float] = None,
    ) -> List[ScoredChunk]:
        """
        Execute hybrid search and merge dense and sparse results via RRF.
        
        Args:
            query: User search query string
            top_k: Number of final fused results to return
            doc_filter: Optional document filename filter
            min_dense_similarity: Minimum cosine similarity threshold (defaults to config)
            
        Returns:
            List of ScoredChunk ordered by descending RRF score.
        """
        threshold = min_dense_similarity if min_dense_similarity is not None else self.similarity_threshold

        # 1. Dense Vector Search (retrieve 2x candidates)
        query_vector = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k * 3,
            doc_filter=doc_filter,
            min_score=threshold,
        )

        # 2. Sparse BM25 Search (retrieve 2x candidates)
        sparse_results = self.bm25_engine.search(
            query=query,
            top_k=top_k * 3,
            doc_filter=doc_filter,
        )

        # 3. Reciprocal Rank Fusion
        # Mapping: chunk_id -> ScoredChunk accumulator
        chunk_map: Dict[str, DocumentChunk] = {}
        dense_ranks: Dict[str, Tuple[int, float]] = {}
        sparse_ranks: Dict[str, Tuple[int, float]] = {}

        for rank, (chunk, score) in enumerate(dense_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            dense_ranks[chunk.chunk_id] = (rank, score)

        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            chunk_map[chunk.chunk_id] = chunk
            sparse_ranks[chunk.chunk_id] = (rank, score)

        if not chunk_map:
            logger.info(f"Hybrid search returned 0 candidates above threshold ({threshold}) for query: {query}")
            return []

        # Calculate RRF score for each unique candidate chunk
        scored_candidates: List[ScoredChunk] = []
        for chunk_id, chunk in chunk_map.items():
            rrf_score = 0.0

            d_rank, d_score = dense_ranks.get(chunk_id, (None, None))
            if d_rank is not None:
                rrf_score += self.dense_weight * (1.0 / (self.rrf_k + d_rank))

            s_rank, s_score = sparse_ranks.get(chunk_id, (None, None))
            if s_rank is not None:
                rrf_score += self.sparse_weight * (1.0 / (self.rrf_k + s_rank))

            scored_candidates.append(
                ScoredChunk(
                    chunk=chunk,
                    rrf_score=rrf_score,
                    dense_score=d_score,
                    sparse_score=s_score,
                    dense_rank=d_rank,
                    sparse_rank=s_rank,
                )
            )

        # Sort descending by RRF score
        scored_candidates.sort(key=lambda x: x.rrf_score, reverse=True)
        return scored_candidates[:top_k]
