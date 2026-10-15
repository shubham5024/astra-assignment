"""Prompt templates and context formatting for grounded RAG generation."""

from typing import List
from src.search.hybrid import ScoredChunk

RAG_SYSTEM_PROMPT = """You are an expert AI knowledge assistant built to answer technical questions with absolute factual precision based EXCLUSIVELY on the provided reference context.

CRITICAL INSTRUCTIONS:
1. Grounding: Answer the question using ONLY the provided context snippets. Do NOT assume, extrapolate, or use outside knowledge.
2. Citations: Every factual claim must be backed by a precise citation tag in the exact format: [Source: <document_name>, Page: <page_number>].
3. Multiple Sources: When combining information across multiple pages or documents, cite each source immediately following the relevant sentence or bullet point.
4. Refusal Rule: If the provided context does not contain sufficient facts to answer the question, or if the question is out of scope of the documents, you MUST explicitly state:
"I do not have sufficient information in the provided knowledge corpus to answer this question."
Do not attempt to guess or invent answers.
5. Tone: Be concise, clear, and direct. Use structured bullet points where appropriate."""

RAG_USER_TEMPLATE = """CONTEXT SNIPPETS:
---------------------
{context_block}
---------------------

USER QUESTION:
{question}

Provide a comprehensive, strictly grounded answer with citations [Source: <document_name>, Page: <page_number>]:"""


def format_context_block(chunks: List[ScoredChunk]) -> str:
    """Format retrieved chunks into structured context with citation headers."""
    if not chunks:
        return "No relevant context found."

    formatted_entries = []
    for idx, scored in enumerate(chunks, start=1):
        chunk = scored.chunk
        header = f"--- [Snippet {idx}] {chunk.to_citation_header()} (Score: {scored.rrf_score:.4f}) ---"
        body = chunk.text.strip()
        formatted_entries.append(f"{header}\n{body}")

    return "\n\n".join(formatted_entries)
