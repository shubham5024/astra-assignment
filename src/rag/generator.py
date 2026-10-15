"""LLM answer generation and citation extraction module."""

from typing import List, Dict, Any, Optional
import time
import re
import logging
from pydantic import BaseModel, Field
from openai import OpenAI

from src.rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE, format_context_block
from src.search.hybrid import ScoredChunk
from src.config import settings

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    """Structured citation metadata for an answer reference."""
    document_name: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="1-indexed source page number")
    raw_tag: str = Field(..., description="Raw citation tag formatted in text")


class GenerationResult(BaseModel):
    """Result of LLM answer generation."""
    answer: str = Field(..., description="Generated answer text")
    citations: List[Citation] = Field(default_factory=list, description="Extracted unique citations")
    model_name: str = Field(..., description="LLM model used")
    is_refusal: bool = Field(default=False, description="Whether the response is a grounded refusal")
    latency_seconds: float = Field(..., description="Generation execution latency")


class LLMGenerator:
    """Manages chat completions and citation validation with fully self-contained offline generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.base_url = base_url or settings.OPENAI_BASE_URL
        self.model_name = model_name or settings.LLM_MODEL
        self.temperature = temperature
        self._client: Optional[OpenAI] = None

        if self.api_key and self.api_key.strip() and self.api_key != "your_openai_api_key_here":
            try:
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for generator: {e}")

    def generate(
        self,
        question: str,
        retrieved_chunks: List[ScoredChunk],
        system_prompt_override: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate a grounded answer for the user question given retrieved context snippets.
        """
        start_time = time.time()

        if not retrieved_chunks:
            refusal_text = (
                "I do not have sufficient information in the provided knowledge corpus to answer this question."
            )
            return GenerationResult(
                answer=refusal_text,
                citations=[],
                model_name=self.model_name if self._client else "deterministic-offline-generator",
                is_refusal=True,
                latency_seconds=round(time.time() - start_time, 4),
            )

        context_block = format_context_block(retrieved_chunks)
        user_prompt = RAG_USER_TEMPLATE.format(context_block=context_block, question=question)
        system_prompt = system_prompt_override or RAG_SYSTEM_PROMPT

        if self._client is not None:
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                )
                raw_answer = response.choices[0].message.content or ""
                active_model = self.model_name
            except Exception as e:
                logger.error(f"OpenAI completion call failed: {e}. Falling back to deterministic context synthesis.")
                raw_answer = self._offline_synthesis(question, retrieved_chunks)
                active_model = "deterministic-offline-generator"
        else:
            raw_answer = self._offline_synthesis(question, retrieved_chunks)
            active_model = "deterministic-offline-generator"

        citations = self.extract_citations(raw_answer, retrieved_chunks)
        is_refusal = (
            "do not have sufficient information" in raw_answer.lower()
            or "not found in the provided" in raw_answer.lower()
        )

        return GenerationResult(
            answer=raw_answer,
            citations=citations,
            model_name=active_model,
            is_refusal=is_refusal,
            latency_seconds=round(time.time() - start_time, 4),
        )

    def extract_citations(self, text: str, chunks: List[ScoredChunk]) -> List[Citation]:
        """Extract and validate citations from text."""
        pattern = re.compile(r"\[Source:\s*([^,\]]+),\s*Page:\s*(\d+)\]", re.IGNORECASE)
        matches = pattern.findall(text)

        extracted: List[Citation] = []
        seen = set()

        for doc_name, page_str in matches:
            doc_clean = doc_name.strip()
            page_num = int(page_str)
            key = (doc_clean.lower(), page_num)
            if key not in seen:
                seen.add(key)
                extracted.append(
                    Citation(
                        document_name=doc_clean,
                        page_number=page_num,
                        raw_tag=f"[Source: {doc_clean}, Page: {page_num}]",
                    )
                )

        if not extracted and chunks and not ("do not have sufficient information" in text.lower()):
            for sc in chunks[:3]:
                c = sc.chunk
                key = (c.document_name.lower(), c.page_number)
                if key not in seen:
                    seen.add(key)
                    extracted.append(
                        Citation(
                            document_name=c.document_name,
                            page_number=c.page_number,
                            raw_tag=c.to_citation_header(),
                        )
                    )

        return extracted

    def _offline_synthesis(self, question: str, chunks: List[ScoredChunk]) -> str:
        """Deterministic extractive synthesis for offline zero-LLM execution."""
        if not chunks:
            return "I do not have sufficient information in the provided knowledge corpus to answer this question."

        # Extract top scoring sentences matching question keywords
        q_words = set(re.findall(r"\w+", question.lower())) - {
            "what", "is", "are", "the", "in", "for", "to", "and", "of", "how", "why", "which"
        }

        extracted_points = []
        seen_points = set()

        for sc in chunks[:3]:
            chunk = sc.chunk
            paragraphs = [p.strip() for p in chunk.text.split("\n") if len(p.strip()) > 20]
            
            for p in paragraphs:
                p_lower = p.lower()
                overlap = sum(1 for w in q_words if w in p_lower)
                if overlap > 0 and p not in seen_points:
                    seen_points.add(p)
                    # Clean snippet
                    clean_p = p.replace("•", "").strip()
                    citation_tag = f"[Source: {chunk.document_name}, Page: {chunk.page_number}]"
                    extracted_points.append(f"- {clean_p} {citation_tag}")
                    if len(extracted_points) >= 4:
                        break
            if len(extracted_points) >= 4:
                break

        if not extracted_points:
            top_chunk = chunks[0].chunk
            citation_tag = f"[Source: {top_chunk.document_name}, Page: {top_chunk.page_number}]"
            summary_snippet = top_chunk.text[:400].strip().replace("\n", " ")
            extracted_points.append(f"{summary_snippet}... {citation_tag}")

        top_doc = chunks[0].chunk.document_name
        top_page = chunks[0].chunk.page_number
        lines = [
            f"Based on the reference documentation in **{top_doc}** (Page {top_page}):",
            "",
            "\n".join(extracted_points),
        ]
        return "\n".join(lines)
