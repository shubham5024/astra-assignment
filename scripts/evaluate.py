"""CLI script to run automated benchmark evaluation over the 15-question evaluation suite."""

import sys
import logging
from rich.console import Console
from rich.table import Table

from src.api.app import get_app_services
from src.evaluation.evaluator import AssistantEvaluator

logging.basicConfig(level=logging.WARNING)
console = Console()


def main():
    console.print("\n[bold cyan]🧪 Running Automated Evaluation Benchmark Suite (15 Test Cases)...[/bold cyan]\n")

    services = get_app_services()
    if not services["vector_store"].chunks:
        console.print("[bold red]❌ Storage index is empty. Please run 'poetry run python scripts/ingest.py' first.[/bold red]")
        sys.exit(1)

    evaluator = AssistantEvaluator(
        router=services["router"],
        simple_rag=services["simple_rag"],
        agent_orchestrator=services["agent_orchestrator"],
        hybrid_searcher=services["hybrid_searcher"],
    )

    report = evaluator.run_benchmark()

    # Results Table
    table = Table(title="Test Case Breakdown", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=8)
    table.add_column("Category", style="cyan", width=22)
    table.add_column("Workflow", width=18)
    table.add_column("Route OK", justify="center", width=10)
    table.add_column("Hit / Refuse", justify="center", width=12)
    table.add_column("Citations", justify="center", width=10)
    table.add_column("Latency (s)", justify="right", width=11)

    for r in report.results:
        route_mark = "[green]✓[/green]" if r.routing_correct else "[red]✗[/red]"
        hit_mark = "[green]✓[/green]" if (r.refusal_correct if "refusal" in r.category else r.retrieval_hit) else "[red]✗[/red]"
        cit_mark = "[green]✓[/green]" if r.has_valid_citations else ("[dim]-[/dim]" if "refusal" in r.category else "[yellow]?[/yellow]")

        table.add_row(
            r.id,
            r.category,
            r.selected_workflow,
            route_mark,
            hit_mark,
            cit_mark,
            f"{r.latency_seconds:.3f}",
        )

    console.print(table)

    # Metrics Summary
    summary_table = Table(title="Aggregate Performance Metrics", show_header=True, header_style="bold green")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Score", justify="right", style="cyan")

    summary_table.add_row("Total Test Cases", str(report.total_queries))
    summary_table.add_row("Intent Routing Precision", f"{report.routing_precision}%")
    summary_table.add_row("Retrieval Hit Rate@5", f"{report.retrieval_hit_rate}%")
    summary_table.add_row("Out-of-Corpus Refusal Accuracy", f"{report.refusal_accuracy}%")
    summary_table.add_row("Citation Grounding Coverage", f"{report.citation_coverage_rate}%")
    summary_table.add_row("Mean Latency", f"{report.average_latency_seconds}s")

    console.print("\n")
    console.print(summary_table)
    console.print("\n[bold green]✅ Evaluation Completed Successfully![/bold green]\n")


if __name__ == "__main__":
    main()
