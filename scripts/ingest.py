"""CLI script to trigger offline document ingestion and index creation."""

import sys
import logging
from rich.console import Console
from rich.table import Table

from src.ingestion.pipeline import IngestionPipeline
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = Console()


def main():
    console.print("\n[bold cyan]🚀 Starting Knowledge Base Ingestion Pipeline...[/bold cyan]")
    console.print(f"📁 Documents Directory: [yellow]{settings.resolved_docs_dir}[/yellow]")
    console.print(f"💾 Storage Directory:   [yellow]{settings.resolved_storage_dir}[/yellow]\n")

    try:
        pipeline = IngestionPipeline()
        summary = pipeline.run()

        # Display rich summary table
        table = Table(title="Ingestion Summary", show_header=True, header_style="bold magenta")
        table.add_column("Document Name", style="cyan")
        table.add_column("Total Pages", justify="right")
        table.add_column("Text Pages", justify="right")
        table.add_column("Chunks", justify="right")

        for doc_name, stats in summary.document_breakdown.items():
            table.add_row(
                doc_name,
                str(stats["pages"]),
                str(stats["text_pages"]),
                str(stats["chunks"]),
            )

        console.print(table)
        console.print(
            f"\n[bold green]✅ Ingestion Successful![/bold green] "
            f"Indexed {summary.total_documents} documents, {summary.total_pages} pages, {summary.total_chunks} chunks.\n"
        )
    except Exception as e:
        console.print(f"\n[bold red]❌ Ingestion Failed:[/bold red] {e}\n", style="bold red")
        sys.exit(1)


if __name__ == "__main__":
    main()
