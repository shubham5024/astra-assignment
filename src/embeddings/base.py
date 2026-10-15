"""Base embedding interface and local deterministic fallback."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import hashlib


class BaseEmbedder(ABC):
    """Abstract interface for text embedding generation."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embedding vectors for a list of texts."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding vector for a single query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass


class DeterministicFallbackEmbedder(BaseEmbedder):
    """
    Lightweight, reproducible n-gram hashing embedder for testing and fallback
    when external API keys are unavailable. Generates normalized unit vectors.
    """

    def __init__(self, dimension: int = 1536):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _hash_to_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec

        for word in words:
            # Hash word and bigrams
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            vec[idx] += 1.0

        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            h = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            vec[idx] += 1.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = [self._hash_to_vector(t) for t in texts]
        return np.vstack(vectors) if vectors else np.empty((0, self._dim), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self._hash_to_vector(query)
