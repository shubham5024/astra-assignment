"""Semantic-aware recursive text chunker with overlap and metadata attachment."""

from typing import List
import re
from src.ingestion.models import DocumentChunk


class SemanticRecursiveChunker:
    """
    Recursively splits document text into semantic chunks respecting natural
    paragraph, sentence, and phrase boundaries while preserving overlap.
    """

    def __init__(
        self,
        chunk_size_tokens: int = 650,
        chunk_overlap_tokens: int = 100,
        separators: List[str] = None,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        # Character approximations: ~4 characters per token
        self.chunk_size_chars = chunk_size_tokens * 4
        self.overlap_chars = chunk_overlap_tokens * 4
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count based on whitespace & character heuristic."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        # Standard heuristic: max of word-count * 1.3 and char-count / 4
        return max(int(words * 1.3), int(chars / 4.0))

    def chunk_page(
        self,
        text: str,
        document_name: str,
        page_number: int,
        start_chunk_index: int = 0,
    ) -> List[DocumentChunk]:
        """
        Split a page's text into DocumentChunk instances with metadata.
        """
        if not text.strip():
            return []

        raw_chunks = self._recursive_split(text, self.separators)
        merged_chunks = self._merge_with_overlap(raw_chunks)

        chunks: List[DocumentChunk] = []
        for idx, chunk_text in enumerate(merged_chunks):
            chunk_idx = start_chunk_index + idx
            # Clean deterministic ID: document_name_p{page}_c{idx}
            clean_doc_name = re.sub(r"[^a-zA-Z0-9_-]", "_", document_name)
            chunk_id = f"{clean_doc_name}_p{page_number}_c{chunk_idx:03d}"
            
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_idx,
                    text=chunk_text.strip(),
                    token_count=self.estimate_tokens(chunk_text),
                    metadata={
                        "source": document_name,
                        "page": page_number,
                        "chunk_index": chunk_idx,
                    },
                )
            )

        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by trying separators from highest to lowest hierarchy."""
        final_chunks: List[str] = []
        
        if not separators:
            return [text]

        sep = separators[0]
        sub_separators = separators[1:]

        if sep == "":
            return list(text)

        splits = text.split(sep)
        for piece in splits:
            if not piece.strip():
                continue
            if len(piece) <= self.chunk_size_chars:
                final_chunks.append(piece)
            else:
                if sub_separators:
                    sub_pieces = self._recursive_split(piece, sub_separators)
                    final_chunks.extend(sub_pieces)
                else:
                    # Fallback slice
                    for i in range(0, len(piece), self.chunk_size_chars):
                        final_chunks.append(piece[i : i + self.chunk_size_chars])

        return final_chunks

    def _merge_with_overlap(self, pieces: List[str]) -> List[str]:
        """Merge atomic pieces into target chunk size with overlap."""
        if not pieces:
            return []

        merged: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in pieces:
            piece_len = len(piece)
            if current_len + piece_len <= self.chunk_size_chars:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    full_text = " ".join(current_chunk)
                    merged.append(full_text)
                    # Create overlap from end of current_chunk
                    overlap_accum: List[str] = []
                    overlap_len = 0
                    for item in reversed(current_chunk):
                        if overlap_len + len(item) <= self.overlap_chars:
                            overlap_accum.insert(0, item)
                            overlap_len += len(item)
                        else:
                            break
                    current_chunk = overlap_accum + [piece]
                    current_len = sum(len(x) for x in current_chunk)
                else:
                    merged.append(piece)
                    current_chunk = []
                    current_len = 0

        if current_chunk:
            merged.append(" ".join(current_chunk))

        return merged
