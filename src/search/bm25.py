"""Sparse BM25 Indexer and Search Engine for exact keyword and acronym matching."""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import re
import logging
from rank_bm25 import BM25Okapi

from src.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class BM25SearchEngine:
    """Inverted index BM25 search engine with tokenization and persistence."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.chunks: List[DocumentChunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tokenized_corpus: List[List[str]] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens without stopwords."""
        clean_text = re.sub(r"[^\w\s-]", " ", text.lower())
        tokens = clean_text.split()
        return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    def build_index(self, chunks: List[DocumentChunk]) -> None:
        """Build BM25 index from a list of DocumentChunk objects."""
        self.chunks = list(chunks)
        self.tokenized_corpus = [self.tokenize(c.text) for c in self.chunks]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"Built BM25 index for {len(self.chunks)} chunks.")
        else:
            self.bm25 = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search top-k chunks matching the query keywords.
        
        Returns:
            List of (DocumentChunk, bm25_score).
        """
        if self.bm25 is None or not self.chunks:
            return []

        tokenized_query = self.tokenize(query)
        if not tokenized_query:
            return []

        raw_scores = self.bm25.get_scores(tokenized_query)
        
        candidates: List[Tuple[int, float]] = []
        for idx, score in enumerate(raw_scores):
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            if doc_filter and chunk.document_name.lower() != doc_filter.lower():
                continue
            candidates.append((idx, float(score)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        return [(self.chunks[idx], score) for idx, score in top_candidates]

    def save(self, directory: Optional[Path] = None) -> None:
        """Persist BM25 chunks and tokenized corpus to disk."""
        target_dir = Path(directory) if directory else self.storage_dir
        if not target_dir:
            raise ValueError("Storage directory must be specified for save()")

        target_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "chunks": [c.model_dump() for c in self.chunks],
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(target_dir / "bm25_index.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info(f"Persisted BM25 index ({len(self.chunks)} chunks) to {target_dir}")

    def load(self, directory: Optional[Path] = None) -> bool:
        """Load BM25 index from disk."""
        target_dir = Path(directory) if directory else self.storage_dir
        if not target_dir:
            return False

        path = target_dir / "bm25_index.json"
        if not path.exists():
            logger.warning(f"BM25 index file not found at {path}")
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.chunks = [DocumentChunk.model_validate(c) for c in data["chunks"]]
        self.tokenized_corpus = data["tokenized_corpus"]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks from {target_dir}")
        return True
