"""Agent tools for granular multi-step retrieval, comparison, and summarization."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import logging

from src.search.hybrid import HybridSearchEngine, ScoredChunk
from src.rag.generator import Citation

logger = logging.getLogger(__name__)


class ToolExecutionResult(BaseModel):
    """Output from an executed agent tool."""
    tool_name: str = Field(..., description="Name of the executed tool")
    success: bool = Field(default=True, description="Execution status")
    output: str = Field(..., description="Textual result/evidence returned by the tool")
    citations: List[Citation] = Field(default_factory=list, description="Citations gathered during execution")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic data")


class AgentToolRegistry:
    """Registry providing tool definitions and execution handlers for agentic workflows."""

    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return schema descriptors for tools."""
        return [
            {
                "name": "search_knowledge_base",
                "description": "Search the knowledge base using hybrid semantic + keyword search. Can filter by specific document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The specific technical search query"},
                        "doc_filter": {
                            "type": "string",
                            "description": "Optional document filter: 'rag_architecture_patterns.pdf', 'vector_database_comparison.pdf', or 'agentic_ai_frameworks.pdf'",
                        },
                        "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default: 4)"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "compare_documents",
                "description": "Retrieve and compare evidence on a specific topic between two distinct documents in the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "The comparative topic or feature to investigate"},
                        "doc_a": {"type": "string", "description": "First document name (e.g. 'vector_database_comparison.pdf')"},
                        "doc_b": {"type": "string", "description": "Second document name (e.g. 'agentic_ai_frameworks.pdf')"},
                    },
                    "required": ["topic", "doc_a", "doc_b"],
                },
            },
            {
                "name": "summarize_section",
                "description": "Retrieve comprehensive context regarding a specific topic within a single document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_name": {"type": "string", "description": "Document name to search within"},
                        "topic": {"type": "string", "description": "Topic or section to summarize"},
                    },
                    "required": ["doc_name", "topic"],
                },
            },
        ]

    def execute_tool(self, name: str, args: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool by name with provided arguments."""
        if name == "search_knowledge_base":
            return self._search_knowledge_base(
                query=args.get("query", ""),
                doc_filter=args.get("doc_filter"),
                top_k=args.get("top_k", 4),
            )
        elif name == "compare_documents":
            return self._compare_documents(
                topic=args.get("topic", ""),
                doc_a=args.get("doc_a", ""),
                doc_b=args.get("doc_b", ""),
            )
        elif name == "summarize_section":
            return self._summarize_section(
                doc_name=args.get("doc_name", ""),
                topic=args.get("topic", ""),
            )
        else:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                output=f"Error: Unknown tool '{name}'. Available tools: search_knowledge_base, compare_documents, summarize_section.",
            )

    def _search_knowledge_base(
        self, query: str, doc_filter: Optional[str] = None, top_k: int = 4
    ) -> ToolExecutionResult:
        scored = self.search_engine.search(query=query, top_k=top_k, doc_filter=doc_filter)
        if not scored:
            return ToolExecutionResult(
                tool_name="search_knowledge_base",
                output=f"No relevant context found for query: '{query}' (filter: {doc_filter}).",
            )

        citations: List[Citation] = []
        snippets: List[str] = []
        for s in scored:
            chunk = s.chunk
            citations.append(
                Citation(
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    raw_tag=chunk.to_citation_header(),
                )
            )
            snippets.append(f"{chunk.to_citation_header()}:\n{chunk.text.strip()}")

        return ToolExecutionResult(
            tool_name="search_knowledge_base",
            output="\n\n".join(snippets),
            citations=citations,
            metadata={"retrieved_count": len(scored)},
        )

    def _compare_documents(self, topic: str, doc_a: str, doc_b: str) -> ToolExecutionResult:
        res_a = self.search_engine.search(query=topic, top_k=3, doc_filter=doc_a)
        res_b = self.search_engine.search(query=topic, top_k=3, doc_filter=doc_b)

        citations: List[Citation] = []
        snippets_a: List[str] = []
        for s in res_a:
            citations.append(
                Citation(document_name=s.chunk.document_name, page_number=s.chunk.page_number, raw_tag=s.chunk.to_citation_header())
            )
            snippets_a.append(f"{s.chunk.to_citation_header()}: {s.chunk.text.strip()}")

        snippets_b: List[str] = []
        for s in res_b:
            citations.append(
                Citation(document_name=s.chunk.document_name, page_number=s.chunk.page_number, raw_tag=s.chunk.to_citation_header())
            )
            snippets_b.append(f"{s.chunk.to_citation_header()}: {s.chunk.text.strip()}")

        combined_output = (
            f"=== EVIDENCE FROM {doc_a} on '{topic}' ===\n"
            + ("\n\n".join(snippets_a) if snippets_a else "No direct matches.")
            + f"\n\n=== EVIDENCE FROM {doc_b} on '{topic}' ===\n"
            + ("\n\n".join(snippets_b) if snippets_b else "No direct matches.")
        )

        return ToolExecutionResult(
            tool_name="compare_documents",
            output=combined_output,
            citations=citations,
            metadata={"doc_a_count": len(res_a), "doc_b_count": len(res_b)},
        )

    def _summarize_section(self, doc_name: str, topic: str) -> ToolExecutionResult:
        res = self.search_engine.search(query=topic, top_k=4, doc_filter=doc_name)
        if not res:
            return ToolExecutionResult(
                tool_name="summarize_section",
                output=f"No matching sections found for topic '{topic}' in document '{doc_name}'.",
            )

        citations = [
            Citation(document_name=s.chunk.document_name, page_number=s.chunk.page_number, raw_tag=s.chunk.to_citation_header())
            for s in res
        ]
        snippets = [f"{s.chunk.to_citation_header()}: {s.chunk.text.strip()}" for s in res]

        return ToolExecutionResult(
            tool_name="summarize_section",
            output=f"=== SECTION CONTEXT FROM {doc_name} for '{topic}' ===\n" + "\n\n".join(snippets),
            citations=citations,
            metadata={"retrieved_count": len(res)},
        )
