"""Automated Evaluation Engine for RAG, Agentic workflows, retrieval precision, and grounding."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import time
import logging
from pydantic import BaseModel, Field

from src.agent.router import QueryRouter
from src.rag.pipeline import SimpleRAGPipeline
from src.agent.orchestrator import AgentOrchestrator
from src.search.hybrid import HybridSearchEngine
from src.config import settings

logger = logging.getLogger(__name__)


class BenchmarkItem(BaseModel):
    """Single benchmark evaluation test case."""
    id: str
    category: str
    question: str
    expected_document: Optional[str] = None
    expected_workflow: str
    must_contain: List[str] = Field(default_factory=list)
    should_refuse: bool = False


class EvalCaseResult(BaseModel):
    """Evaluation result for an individual benchmark item."""
    id: str
    category: str
    question: str
    selected_workflow: str
    expected_workflow: str
    routing_correct: bool
    retrieval_hit: bool
    refusal_correct: bool
    has_valid_citations: bool
    latency_seconds: float
    answer_preview: str


class EvaluationReport(BaseModel):
    """Aggregated evaluation metrics across the benchmark suite."""
    total_queries: int
    routing_precision: float
    retrieval_hit_rate: float
    refusal_accuracy: float
    citation_coverage_rate: float
    average_latency_seconds: float
    category_breakdown: Dict[str, Dict[str, Any]]
    results: List[EvalCaseResult]


class AssistantEvaluator:
    """Runs automated benchmark evaluation and computes performance metrics."""

    def __init__(
        self,
        router: QueryRouter,
        simple_rag: SimpleRAGPipeline,
        agent_orchestrator: AgentOrchestrator,
        hybrid_searcher: HybridSearchEngine,
    ):
        self.router = router
        self.simple_rag = simple_rag
        self.agent_orchestrator = agent_orchestrator
        self.hybrid_searcher = hybrid_searcher

    def run_benchmark(self, benchmark_file: Optional[Path] = None) -> EvaluationReport:
        """Run evaluation over benchmark cases and calculate metrics."""
        path = benchmark_file or (Path(__file__).parent / "benchmark.json")
        with open(path, "r", encoding="utf-8") as f:
            raw_cases = json.load(f)

        cases = [BenchmarkItem.model_validate(c) for c in raw_cases]
        results: List[EvalCaseResult] = []

        total_routing_correct = 0
        total_retrieval_hits = 0
        in_corpus_count = 0
        refusal_tests_count = 0
        refusal_correct_count = 0
        citation_valid_count = 0
        answered_count = 0
        latencies: List[float] = []

        category_stats: Dict[str, Dict[str, Any]] = {}

        for item in cases:
            start_t = time.time()
            decision = self.router.route(item.question)
            selected_workflow = decision.workflow

            routing_correct = (selected_workflow == item.expected_workflow)
            if routing_correct:
                total_routing_correct += 1

            # Execute workflow
            if selected_workflow == "DIRECT_REFUSAL":
                answer = "I do not have sufficient information in the provided knowledge corpus to answer this question."
                citations = []
                is_refusal = True
                retrieval_hit = True if item.should_refuse else False
            elif selected_workflow == "AGENTIC_WORKFLOW":
                res = self.agent_orchestrator.run(item.question)
                answer = res.answer
                citations = res.citations
                is_refusal = res.is_refusal
                retrieval_hit = any(item.expected_document in c.document_name for c in citations) if item.expected_document else True
            else:  # SIMPLE_RAG
                rag_res = self.simple_rag.answer_query(item.question)
                answer = rag_res.answer
                citations = rag_res.citations
                is_refusal = rag_res.is_refusal
                retrieval_hit = any(item.expected_document in c.document_name for c in citations) if item.expected_document else True

            elapsed = round(time.time() - start_t, 4)
            latencies.append(elapsed)

            # Refusal validation
            if item.should_refuse:
                refusal_tests_count += 1
                refusal_ok = is_refusal or ("not have sufficient information" in answer.lower())
                if refusal_ok:
                    refusal_correct_count += 1
            else:
                refusal_ok = not is_refusal
                in_corpus_count += 1
                if retrieval_hit:
                    total_retrieval_hits += 1
                answered_count += 1
                if citations:
                    citation_valid_count += 1

            has_citations = len(citations) > 0

            res_obj = EvalCaseResult(
                id=item.id,
                category=item.category,
                question=item.question,
                selected_workflow=selected_workflow,
                expected_workflow=item.expected_workflow,
                routing_correct=routing_correct,
                retrieval_hit=retrieval_hit,
                refusal_correct=refusal_ok,
                has_valid_citations=has_citations,
                latency_seconds=elapsed,
                answer_preview=answer[:120].replace("\n", " ") + "...",
            )
            results.append(res_obj)

            # Category aggregation
            cat = item.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "routing_correct": 0, "success": 0}
            category_stats[cat]["total"] += 1
            if routing_correct:
                category_stats[cat]["routing_correct"] += 1
            if (refusal_ok if item.should_refuse else retrieval_hit):
                category_stats[cat]["success"] += 1

        routing_prec = round((total_routing_correct / len(cases)) * 100, 1)
        hit_rate = round((total_retrieval_hits / max(1, in_corpus_count)) * 100, 1)
        refusal_acc = round((refusal_correct_count / max(1, refusal_tests_count)) * 100, 1)
        citation_cov = round((citation_valid_count / max(1, answered_count)) * 100, 1)
        avg_latency = round(sum(latencies) / len(latencies), 3)

        return EvaluationReport(
            total_queries=len(cases),
            routing_precision=routing_prec,
            retrieval_hit_rate=hit_rate,
            refusal_accuracy=refusal_acc,
            citation_coverage_rate=citation_cov,
            average_latency_seconds=avg_latency,
            category_breakdown=category_stats,
            results=results,
        )
