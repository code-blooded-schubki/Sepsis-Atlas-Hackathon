"""
run_pipeline.py — Main entry point for the Sepsis Atlas extraction pipeline.

Usage:
    python run_pipeline.py                        # process all PDFs in data/papers/
    python run_pipeline.py --file my_paper.pdf    # process a single PDF
    python run_pipeline.py --export-csv           # export DB to CSV (no re-extraction)

What it does for each PDF:
  1. Extract text from the PDF
  2. Send to Claude for structured extraction
  3. Validate + score the result
  4. Save to SQLite database
  5. Save individual JSON to data/outputs/
  6. Print a summary to console
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

import config
from pipeline.pdf_reader import extract_text, get_relevant_chunk
from pipeline.extractor import extract_paper
from pipeline.validator import validate_paper, filter_low_confidence_fields, summarise_extraction
from utils.db import init_db, paper_exists, save_paper, export_to_csv
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


def process_pdf(pdf_path: Path) -> bool:
    """
    Run the full pipeline on a single PDF.
    Returns True on success, False on failure.
    """
    paper_id = pdf_path.stem

    # Skip if already processed
    if config.SKIP_ALREADY_EXTRACTED and paper_exists(paper_id):
        logger.info(f"Skipping {pdf_path.name} (already in database)")
        return True

    try:
        # Step 1: Extract text
        full_text = extract_text(pdf_path)
        chunk = get_relevant_chunk(full_text, max_chars=config.CHUNK_SIZE)

        # Step 2: LLM extraction
        paper = extract_paper(chunk, pdf_path.name)

        # Step 3: Validate
        is_valid, warnings = validate_paper(paper)
        if warnings:
            for w in warnings:
                logger.warning(f"  ⚠ {w}")

        # Step 4: Filter low-confidence fields
        paper = filter_low_confidence_fields(paper, min_confidence=config.MIN_CONFIDENCE)
        paper.compute_overall_confidence()

        # Step 5: Save to DB
        save_paper(paper)

        # Step 6: Save JSON output
        json_path = config.OUTPUT_DIR / f"{paper_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(paper.model_dump_json(indent=2))

        # Step 7: Print summary
        console.print(summarise_extraction(paper))
        if not is_valid:
            console.print(f"  [yellow]⚠ Flagged for review[/yellow]")

        return True

    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
        return False


def print_results_table() -> None:
    """Print a rich table of all extracted papers in the DB."""
    import pandas as pd
    from utils.db import load_all_papers

    df = load_all_papers()
    if df.empty:
        console.print("[yellow]No papers in database yet.[/yellow]")
        return

    table = Table(title=f"Sepsis Atlas — {len(df)} papers extracted", show_lines=True)
    cols = ["pdf_filename", "meta_title", "meta_year", "pop_sample_size",
            "out_mortality", "sep_definition", "overall_confidence"]
    labels = ["File", "Title", "Year", "N", "Mortality", "Sepsis def.", "Confidence"]

    for col, label in zip(cols, labels):
        table.add_column(label, max_width=30, overflow="fold")

    for _, row in df.iterrows():
        table.add_row(*[
            str(row.get(c, ""))[:60] if row.get(c) else "[dim]N/A[/dim]"
            for c in cols
        ])

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Sepsis Atlas extraction pipeline")
    parser.add_argument("--file", type=str, help="Process a single PDF file")
    parser.add_argument("--export-csv", action="store_true", help="Export DB to CSV and exit")
    parser.add_argument("--show-table", action="store_true", help="Print results table and exit")
    args = parser.parse_args()

    # Init DB
    init_db()

    # Export-only mode
    if args.export_csv:
        path = export_to_csv()
        console.print(f"[green]✓ Exported to {path}[/green]")
        return

    if args.show_table:
        print_results_table()
        return

    # Collect PDFs to process
    if args.file:
        pdf_paths = [Path(args.file)]
        if not pdf_paths[0].exists():
            console.print(f"[red]File not found: {args.file}[/red]")
            sys.exit(1)
    else:
        pdf_paths = sorted(config.PAPERS_DIR.glob("*.pdf"))
        if not pdf_paths:
            console.print(
                f"[yellow]No PDFs found in {config.PAPERS_DIR}. "
                f"Drop some papers there and re-run.[/yellow]"
            )
            sys.exit(0)

    console.print(f"\n[bold]Sepsis Atlas Pipeline[/bold] — processing {len(pdf_paths)} paper(s)\n")

    # Process each PDF
    success, failed = 0, 0
    for pdf_path in tqdm(pdf_paths, desc="Extracting", unit="paper"):
        ok = process_pdf(pdf_path)
        if ok:
            success += 1
        else:
            failed += 1

    console.print(f"\n[green]✓ {success} succeeded[/green]  [red]✗ {failed} failed[/red]\n")

    # Auto-export CSV
    csv_path = export_to_csv()
    console.print(f"[green]CSV exported → {csv_path}[/green]")

    # Show results table
    print_results_table()


if __name__ == "__main__":
    main()
