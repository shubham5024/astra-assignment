"""Persistent Dense Vector Store implementation with cosine similarity and metadata filtering."""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import logging
import numpy as np

from src.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)


class DenseVectorStore:
    """
    In-memory normalized vector index with disk serialization and metadata filtering.
    Computes exact Cosine Similarity with dot-product operations over L2-normalized embeddings.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.chunks: List[DocumentChunk] = []
        self.chunk_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._id_to_idx: Dict[str, int] = {}

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Add chunks and their corresponding embedding vectors to the index."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embedding rows.")

        if not chunks:
            return

        # Ensure embeddings are normalized float32
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embeddings = (embeddings / norms).astype(np.float32)

        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        for i, chunk in enumerate(chunks):
            self.chunk_ids.append(chunk.chunk_id)
            self._id_to_idx[chunk.chunk_id] = start_idx + i

        if self.embeddings is None:
            self.embeddings = normalized_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, normalized_embeddings])

        logger.info(f"Added {len(chunks)} chunks to vector store. Total chunks: {len(self.chunks)}.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        min_score: float = -1.0,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for top-k most similar chunks.
        
        Returns:
            List of (DocumentChunk, cosine_similarity_score).
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # Normalize query vector
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 0:
            q_vec = (query_vector / q_norm).astype(np.float32)
        else:
            q_vec = query_vector.astype(np.float32)

        # Dot product gives cosine similarity since vectors are normalized
        scores = np.dot(self.embeddings, q_vec)

        # Apply metadata filter if specified
        candidates: List[Tuple[int, float]] = []
        for idx, score in enumerate(scores):
            if score < min_score:
                continue
            chunk = self.chunks[idx]
            if doc_filter and chunk.document_name.lower() != doc_filter.lower():
                continue
            candidates.append((idx, float(score)))

        # Sort descending by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        return [(self.chunks[idx], score) for idx, score in top_candidates]

    def save(self, directory: Optional[Path] = None) -> None:
        """Persist vector store to disk."""
        target_dir = Path(directory) if directory else self.storage_dir
        if not target_dir:
            raise ValueError("Storage directory must be specified for save()")

        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(str(target_dir / "dense_embeddings.npy"), self.embeddings)

        # Save metadata and chunk records
        chunks_data = [chunk.model_dump() for chunk in self.chunks]
        with open(target_dir / "dense_chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Persisted vector store ({len(self.chunks)} chunks) to {target_dir}")

    def load(self, directory: Optional[Path] = None) -> bool:
        """Load vector store from disk."""
        target_dir = Path(directory) if directory else self.storage_dir
        if not target_dir:
            return False

        emb_path = target_dir / "dense_embeddings.npy"
        chunk_path = target_dir / "dense_chunks.json"

        if not emb_path.exists() or not chunk_path.exists():
            logger.warning(f"Vector store files not found in {target_dir}")
            return False

        self.embeddings = np.load(str(emb_path))
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        self.chunks = [DocumentChunk.model_validate(item) for item in chunks_data]
        self.chunk_ids = [c.chunk_id for c in self.chunks]
        self._id_to_idx = {c.chunk_id: i for i, c in enumerate(self.chunks)}

        logger.info(f"Loaded vector store with {len(self.chunks)} chunks from {target_dir}")
        return True

    def count(self) -> int:
        return len(self.chunks)
