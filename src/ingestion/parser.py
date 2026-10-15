"""PDF parsing and text extraction module."""

from pathlib import Path
from typing import List, Dict, Any
import logging
from pypdf import PdfReader

from src.ingestion.models import DocumentMetadata

logger = logging.getLogger(__name__)


class PDFParser:
    """Extracts text page-by-page from PDF files with metadata preservation."""

    def __init__(self, clean_whitespace: bool = True):
        self.clean_whitespace = clean_whitespace

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a single PDF file and extract pages with 1-indexed numbering.
        
        Returns:
            Dict containing 'metadata' (DocumentMetadata) and 'pages' (List of Dict[page_number, text]).
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF document not found at {file_path}")

        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        pages_content: List[Dict[str, Any]] = []

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            if self.clean_whitespace:
                text = self._clean_text(raw_text)
            else:
                text = raw_text

            if text.strip():
                pages_content.append({
                    "page_number": page_num,
                    "text": text,
                })

        # Infer title from filename or first page
        title = path.stem.replace("_", " ").title()
        metadata = DocumentMetadata(
            document_name=path.name,
            file_path=str(path),
            total_pages=total_pages,
            document_title=title,
        )

        logger.info(f"Parsed {path.name}: {len(pages_content)} pages with text out of {total_pages} total pages.")
        return {
            "metadata": metadata,
            "pages": pages_content,
        }

    def _clean_text(self, text: str) -> str:
        """Normalizes excessive whitespace while preserving paragraph breaks."""
        lines = [line.strip() for line in text.splitlines()]
        cleaned = "\n".join([line for line in lines if line])
        # Replace multiple spaces with single space
        import re
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        return cleaned
