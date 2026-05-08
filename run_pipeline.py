"""
run_pipeline.py — Main entry point.

Usage:
    python run_pipeline.py                     # process all PDFs in data/papers/
    python run_pipeline.py --file paper.pdf    # single PDF
    python run_pipeline.py --export-csv        # export cohorts CSV
    python run_pipeline.py --show-table        # print results table
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

import config
from pipeline.pdf_reader import extract_text, get_relevant_chunk
from pipeline.extractor import extract_paper
from pipeline.validator import validate_paper, filter_low_confidence_fields, summarise_extraction
from pipeline.verifier import verify_paper
from utils.db import init_db, paper_exists, save_paper, save_cohorts, save_findings, export_to_csv, load_all_papers
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


def process_pdf(pdf_path: Path) -> bool:
    paper_id = pdf_path.stem

    if config.SKIP_ALREADY_EXTRACTED and paper_exists(paper_id):
        logger.info(f"Skipping {pdf_path.name} (already in database)")
        return True

    try:
        # 1. Extract text
        full_text = extract_text(pdf_path)
        llm_text  = get_relevant_chunk(full_text, max_chars=config.CHUNK_SIZE)

        # 2. LLM structured extraction
        paper = extract_paper(llm_text, pdf_path.name)

        # 3. Validate + filter
        is_valid, warnings = validate_paper(paper)
        for w in warnings:
            logger.warning(f"  {w}")
        if not is_valid:
            console.print(f"  [yellow]⚠ {pdf_path.name} flagged for review[/yellow]")

        paper = filter_low_confidence_fields(paper, min_confidence=config.MIN_CONFIDENCE)
        paper.compute_overall_confidence()

        # 4. Verifiability — checked against raw PDF text, no LLM involved
        v = verify_paper(paper, full_text)
        paper.overall_verifiability = v["overall_verifiability"]
        logger.info(f"{pdf_path.name} — verifiability: {paper.overall_verifiability:.2f}")

        # 5. Save paper + cohorts + findings
        save_paper(paper)
        save_cohorts(paper)
        save_findings(paper)

        # 5. Save JSON
        json_path = config.OUTPUT_DIR / f"{paper_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(paper.model_dump_json(indent=2))

        n_cohorts = len(paper.cohorts)
        console.print(summarise_extraction(paper))
        console.print(f"  [cyan]↳ {n_cohorts} cohort(s) | confidence: {paper.overall_confidence:.2f} | verifiability: {paper.overall_verifiability:.2f}[/cyan]")
        return True

    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
        return False


def print_results_table() -> None:
    df = load_all_papers()
    if df.empty:
        console.print("[yellow]No papers in database yet.[/yellow]")
        return
    table = Table(title=f"Sepsis Atlas — {len(df)} papers", show_lines=True)
    cols   = ["pdf_filename", "meta_title", "meta_year", "meta_study_design", "sep_definition", "overall_confidence", "overall_verifiability"]
    labels = ["File", "Title", "Year", "Design", "Sepsis def.", "Conf.", "Verify"]
    for label in labels:
        table.add_column(label, max_width=30, overflow="fold")
    for _, row in df.iterrows():
        table.add_row(*[str(row.get(c, ""))[:60] if row.get(c) else "[dim]N/A[/dim]" for c in cols])
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Sepsis Atlas pipeline")
    parser.add_argument("--file",       type=str)
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--show-table", action="store_true")
    args = parser.parse_args()

    init_db()

    if args.export_csv:
        console.print(f"[green]Exported → {export_to_csv()}[/green]")
        return

    if args.show_table:
        print_results_table()
        return

    if args.file:
        pdf_paths = [Path(args.file)]
        if not pdf_paths[0].exists():
            console.print(f"[red]File not found: {args.file}[/red]")
            sys.exit(1)
    else:
        pdf_paths = sorted(config.PAPERS_DIR.glob("*.pdf"))
        if not pdf_paths:
            console.print(f"[yellow]No PDFs in {config.PAPERS_DIR}[/yellow]")
            sys.exit(0)

    console.print(f"\n[bold]Sepsis Atlas Pipeline[/bold] — {len(pdf_paths)} paper(s)\n")

    success, failed = 0, 0
    for pdf_path in tqdm(pdf_paths, desc="Processing", unit="paper"):
        if process_pdf(pdf_path):
            success += 1
        else:
            failed += 1

    console.print(f"\n[green]✓ {success} succeeded[/green]  [red]✗ {failed} failed[/red]\n")
    console.print(f"[green]CSV → {export_to_csv()}[/green]")
    print_results_table()


if __name__ == "__main__":
    main()
