"""OpenAI embedding model integration."""

from typing import List, Optional
import logging
import numpy as np
from openai import OpenAI

from src.embeddings.base import BaseEmbedder, DeterministicFallbackEmbedder
from src.config import settings

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Generates embeddings using OpenAI text-embedding-3-small (or specified model)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model = model or settings.EMBEDDING_MODEL
        self._dim = dimension or settings.EMBEDDING_DIMENSIONS

        self._fallback_embedder: Optional[DeterministicFallbackEmbedder] = None
        self._client: Optional[OpenAI] = None

        if self.api_key and self.api_key.strip() and self.api_key != "your_openai_api_key_here":
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}. Falling back to deterministic embedder.")
                self._fallback_embedder = DeterministicFallbackEmbedder(dimension=self._dim)
        else:
            logger.info("No valid OpenAI API key found; initializing deterministic offline embedder.")
            self._fallback_embedder = DeterministicFallbackEmbedder(dimension=self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed a list of text strings in batches."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        if self._fallback_embedder is not None:
            return self._fallback_embedder.embed_texts(texts)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace empty strings with a single space
            sanitized_batch = [t if t.strip() else " " for t in batch]
            try:
                response = self._client.embeddings.create(
                    model=self.model,
                    input=sanitized_batch,
                )
                batch_vectors = [d.embedding for d in response.data]
                all_embeddings.extend(batch_vectors)
            except Exception as e:
                logger.error(f"OpenAI embedding batch API call failed: {e}. Using deterministic fallback.")
                if self._fallback_embedder is None:
                    self._fallback_embedder = DeterministicFallbackEmbedder(dimension=self._dim)
                return self._fallback_embedder.embed_texts(texts)

        matrix = np.array(all_embeddings, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        if self._fallback_embedder is not None:
            return self._fallback_embedder.embed_query(query)

        try:
            sanitized = query.strip() if query.strip() else " "
            response = self._client.embeddings.create(
                model=self.model,
                input=[sanitized],
            )
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        except Exception as e:
            logger.error(f"OpenAI embed_query failed: {e}. Using deterministic fallback.")
            if self._fallback_embedder is None:
                self._fallback_embedder = DeterministicFallbackEmbedder(dimension=self._dim)
            return self._fallback_embedder.embed_query(query)
