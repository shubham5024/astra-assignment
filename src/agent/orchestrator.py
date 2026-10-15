"""Multi-Step ReAct Agent Orchestrator for complex reasoning and tool execution."""

from typing import List, Dict, Any, Optional
import time
import json
import re
import logging
from pydantic import BaseModel, Field
from openai import OpenAI

from src.agent.tools import AgentToolRegistry, ToolExecutionResult
from src.rag.generator import Citation, LLMGenerator
from src.config import settings

logger = logging.getLogger(__name__)


class AgentStep(BaseModel):
    """A single reasoning and action step in the agent execution trace."""
    step_number: int = Field(..., description="1-indexed sequence number of the step")
    thought: str = Field(..., description="Agent internal reasoning thought")
    tool_name: Optional[str] = Field(default=None, description="Tool invoked in this step")
    tool_input: Optional[Dict[str, Any]] = Field(default=None, description="Input parameters passed to the tool")
    observation: Optional[str] = Field(default=None, description="Output returned by the tool")
    latency_seconds: float = Field(..., description="Step execution time in seconds")


class AgentResponse(BaseModel):
    """Complete response returned by the Agentic Orchestrator."""
    query: str = Field(..., description="User query")
    answer: str = Field(..., description="Synthesized grounded answer")
    citations: List[Citation] = Field(default_factory=list, description="All citations accumulated across tool calls")
    workflow: str = Field(default="AGENTIC_WORKFLOW", description="Workflow designation")
    steps: List[AgentStep] = Field(default_factory=list, description="Detailed trace of execution steps")
    is_refusal: bool = Field(default=False, description="Whether response is a refusal")
    total_iterations: int = Field(..., description="Total ReAct loops executed")
    latency_seconds: float = Field(..., description="Total execution latency in seconds")


REACT_PROMPT_TEMPLATE = """You are a senior technical AI research agent. Your goal is to provide a comprehensive, strictly grounded answer to the user query using the available technical tools.

AVAILABLE TOOLS:
1. search_knowledge_base: {{"query": "<search query>", "doc_filter": "<optional doc name>", "top_k": 4}}
2. compare_documents: {{"topic": "<topic to compare>", "doc_a": "<doc1>", "doc_b": "<doc2>"}}
3. summarize_section: {{"doc_name": "<doc name>", "topic": "<topic>"}}

DOCUMENT NAMES IN CORPUS:
- 'rag_architecture_patterns.pdf'
- 'vector_database_comparison.pdf'
- 'agentic_ai_frameworks.pdf'

FORMAT INSTRUCTIONS:
To use a tool, respond ONLY with a JSON object in this exact schema:
{{"thought": "<your reasoning for what to investigate next>", "tool": "<tool_name>", "args": {{"arg1": "val1"}}}}

When you have gathered enough factual evidence and are ready to present the final answer, respond ONLY with:
{{"thought": "<reasoning that sufficient evidence is gathered>", "final_answer": "<your complete, structured answer citing sources as [Source: <doc>, Page: <page>]>"}}

PREVIOUS SCRATCHPAD:
{scratchpad}

USER QUERY: {query}
"""


class AgentOrchestrator:
    """ReAct State Machine Orchestrator for complex, multi-step knowledge tasks."""

    def __init__(
        self,
        tool_registry: AgentToolRegistry,
        generator: Optional[LLMGenerator] = None,
        max_iterations: int = None,
    ):
        self.tool_registry = tool_registry
        self.generator = generator or LLMGenerator()
        self.max_iterations = max_iterations or settings.AGENT_MAX_ITERATIONS
        self._client: Optional[OpenAI] = None

        if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.strip() and settings.OPENAI_API_KEY != "your_openai_api_key_here":
            try:
                self._client = OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL)
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client for agent: {e}")

    def run(self, query: str) -> AgentResponse:
        """Execute the multi-step agent reasoning loop."""
        start_time = time.time()
        steps: List[AgentStep] = []
        all_citations: List[Citation] = []
        seen_citations = set()
        scratchpad = ""

        if self._client is not None:
            # LLM ReAct Loop
            for iteration in range(1, self.max_iterations + 1):
                step_start = time.time()
                prompt = REACT_PROMPT_TEMPLATE.format(scratchpad=scratchpad or "None so far.", query=query)

                try:
                    response = self._client.chat.completions.create(
                        model=settings.LLM_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=settings.AGENT_TEMPERATURE,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or "{}"
                    parsed = json.loads(content)
                except Exception as e:
                    logger.error(f"ReAct agent iteration {iteration} failed: {e}")
                    break

                thought = parsed.get("thought", "Analyzing query and formulating plan.")

                if "final_answer" in parsed:
                    # Agent concluded reasoning
                    final_text = parsed["final_answer"]
                    step = AgentStep(
                        step_number=iteration,
                        thought=thought,
                        tool_name="None",
                        tool_input={},
                        observation="Final answer synthesized.",
                        latency_seconds=round(time.time() - step_start, 4),
                    )
                    steps.append(step)
                    
                    # Extract any citations in final answer
                    for cit in self.generator.extract_citations(final_text, []):
                        key = (cit.document_name.lower(), cit.page_number)
                        if key not in seen_citations:
                            seen_citations.add(key)
                            all_citations.append(cit)

                    is_refusal = "do not have sufficient information" in final_text.lower()
                    return AgentResponse(
                        query=query,
                        answer=final_text,
                        citations=all_citations,
                        workflow="AGENTIC_WORKFLOW",
                        steps=steps,
                        is_refusal=is_refusal,
                        total_iterations=iteration,
                        latency_seconds=round(time.time() - start_time, 4),
                    )

                # Tool invocation step
                tool_name = parsed.get("tool")
                tool_args = parsed.get("args", {})

                if not tool_name:
                    logger.warning("No tool specified in agent response. Breaking loop.")
                    break

                tool_res: ToolExecutionResult = self.tool_registry.execute_tool(tool_name, tool_args)
                step_latency = round(time.time() - step_start, 4)

                step = AgentStep(
                    step_number=iteration,
                    thought=thought,
                    tool_name=tool_name,
                    tool_input=tool_args,
                    observation=tool_res.output[:800] + ("..." if len(tool_res.output) > 800 else ""),
                    latency_seconds=step_latency,
                )
                steps.append(step)

                # Collect citations
                for c in tool_res.citations:
                    key = (c.document_name.lower(), c.page_number)
                    if key not in seen_citations:
                        seen_citations.add(key)
                        all_citations.append(c)

                # Append to scratchpad
                scratchpad += f"\n[Step {iteration}]\nThought: {thought}\nAction: {tool_name}({json.dumps(tool_args)})\nObservation: {tool_res.output}\n"

        # Fallback / Deterministic Multi-Step Orchestration
        return self._deterministic_multi_step_run(query, start_time, steps, all_citations, seen_citations)

    def _deterministic_multi_step_run(
        self,
        query: str,
        start_time: float,
        steps: List[AgentStep],
        all_citations: List[Citation],
        seen_citations: set,
    ) -> AgentResponse:
        """Deterministic multi-step decomposition for offline execution / testing."""
        q_lower = query.lower()

        # Step 1: Broad hybrid retrieval across all documents
        step1_start = time.time()
        thought1 = "Decomposing query to search core concepts across entire knowledge base."
        t1_res = self.tool_registry.execute_tool("search_knowledge_base", {"query": query, "top_k": 4})
        
        step1 = AgentStep(
            step_number=len(steps) + 1,
            thought=thought1,
            tool_name="search_knowledge_base",
            tool_input={"query": query, "top_k": 4},
            observation=t1_res.output[:600] + ("..." if len(t1_res.output) > 600 else ""),
            latency_seconds=round(time.time() - step1_start, 4),
        )
        steps.append(step1)
        for c in t1_res.citations:
            key = (c.document_name.lower(), c.page_number)
            if key not in seen_citations:
                seen_citations.add(key)
                all_citations.append(c)

        # Step 2: Targeted comparison or secondary domain query
        step2_start = time.time()
        if "compare" in q_lower or "vs" in q_lower or "difference" in q_lower:
            thought2 = "Executing cross-document comparative analysis between Vector DBs and Agent Frameworks."
            t2_res = self.tool_registry.execute_tool(
                "compare_documents",
                {
                    "topic": query,
                    "doc_a": "vector_database_comparison.pdf",
                    "doc_b": "agentic_ai_frameworks.pdf",
                },
            )
            step2_tool = "compare_documents"
            step2_input = {"topic": query, "doc_a": "vector_database_comparison.pdf", "doc_b": "agentic_ai_frameworks.pdf"}
        else:
            thought2 = "Executing focused section search on architecture patterns."
            t2_res = self.tool_registry.execute_tool(
                "summarize_section",
                {"doc_name": "rag_architecture_patterns.pdf", "topic": query},
            )
            step2_tool = "summarize_section"
            step2_input = {"doc_name": "rag_architecture_patterns.pdf", "topic": query}

        step2 = AgentStep(
            step_number=len(steps) + 1,
            thought=thought2,
            tool_name=step2_tool,
            tool_input=step2_input,
            observation=t2_res.output[:600] + ("..." if len(t2_res.output) > 600 else ""),
            latency_seconds=round(time.time() - step2_start, 4),
        )
        steps.append(step2)
        for c in t2_res.citations:
            key = (c.document_name.lower(), c.page_number)
            if key not in seen_citations:
                seen_citations.add(key)
                all_citations.append(c)

        # Synthesize collected observations
        if not all_citations:
            final_text = "I do not have sufficient information in the provided knowledge corpus to answer this question."
            is_refusal = True
        else:
            citation_tags = " ".join([c.raw_tag for c in all_citations[:4]])
            final_text = (
                f"### Multi-Step Analysis Summary\n\n"
                f"Based on cross-document evidence gathered across {len(all_citations)} cited sections:\n\n"
                f"1. **Core Findings**: The retrieved sources address '{query}' across architectural patterns, database trade-offs, and agent framework capabilities.\n"
                f"2. **Comparative Synthesis**: {t1_res.output[:350].strip()}...\n\n"
                f"**References**: {citation_tags}"
            )
            is_refusal = False

        total_elapsed = round(time.time() - start_time, 4)
        return AgentResponse(
            query=query,
            answer=final_text,
            citations=all_citations,
            workflow="AGENTIC_WORKFLOW",
            steps=steps,
            is_refusal=is_refusal,
            total_iterations=len(steps),
            latency_seconds=total_elapsed,
        )
