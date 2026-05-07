"""
run_pipeline.py — Main entry point.

Usage:
    python run_pipeline.py                     # process all PDFs in data/papers/
    python run_pipeline.py --file paper.pdf    # single PDF
    python run_pipeline.py --export-csv        # export DB to CSV, no re-extraction
    python run_pipeline.py --show-table        # print results table
    python run_pipeline.py --search "query"    # semantic search over chunks
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

import config
from pipeline.pdf_reader import extract_text, get_relevant_chunk, parse_sections, chunk_paper
from pipeline.extractor import extract_paper
from pipeline.validator import validate_paper, filter_low_confidence_fields, summarise_extraction
from utils.db import init_db, paper_exists, save_paper, save_sections, save_chunks, export_to_csv, search_chunks
from utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


def process_pdf(pdf_path: Path) -> bool:
    paper_id = pdf_path.stem

    if config.SKIP_ALREADY_EXTRACTED and paper_exists(paper_id):
        logger.info(f"Skipping {pdf_path.name} (already in database)")
        return True

    try:
        # 1. Extract full text
        full_text = extract_text(pdf_path)

        # 2. Parse sections → save to SQLite
        sections = parse_sections(full_text, paper_id)
        save_sections(sections)
        logger.info(f"{pdf_path.name}: {len(sections)} sections detected")

        # 3. Chunk within sections → save to ChromaDB
        chunks = chunk_paper(sections, paper_id)
        save_chunks(chunks)
        logger.info(f"{pdf_path.name}: {len(chunks)} chunks saved")

        # 4. LLM structured extraction (uses truncated text to fit context)
        llm_text = get_relevant_chunk(full_text, max_chars=config.CHUNK_SIZE)
        paper = extract_paper(llm_text, pdf_path.name)

        # 5. Validate + filter low-confidence
        is_valid, warnings = validate_paper(paper)
        for w in warnings:
            logger.warning(f"  {w}")
        if not is_valid:
            console.print(f"  [yellow]⚠ {pdf_path.name} flagged for review[/yellow]")
        paper = filter_low_confidence_fields(paper, min_confidence=config.MIN_CONFIDENCE)
        paper.compute_overall_confidence()

        # 6. Save structured extraction to SQLite
        save_paper(paper)

        # 7. Save JSON output
        json_path = config.OUTPUT_DIR / f"{paper_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(paper.model_dump_json(indent=2))

        console.print(summarise_extraction(paper))
        return True

    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}", exc_info=True)
        return False


def print_results_table() -> None:
    from utils.db import load_all_papers
    df = load_all_papers()
    if df.empty:
        console.print("[yellow]No papers in database yet.[/yellow]")
        return

    table = Table(title=f"Sepsis Atlas — {len(df)} papers", show_lines=True)
    cols   = ["pdf_filename", "meta_title", "meta_year", "pop_sample_size", "out_mortality", "sep_definition", "overall_confidence"]
    labels = ["File", "Title", "Year", "N", "Mortality", "Sepsis def.", "Conf."]
    for label in labels:
        table.add_column(label, max_width=30, overflow="fold")
    for _, row in df.iterrows():
        table.add_row(*[str(row.get(c, ""))[:60] if row.get(c) else "[dim]N/A[/dim]" for c in cols])
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Sepsis Atlas pipeline")
    parser.add_argument("--file",       type=str, help="Process a single PDF")
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--show-table", action="store_true")
    parser.add_argument("--search",     type=str, help="Semantic search query over chunks")
    parser.add_argument("--section",    type=str, help="Filter --search to a specific section")
    args = parser.parse_args()

    init_db()

    if args.export_csv:
        console.print(f"[green]Exported → {export_to_csv()}[/green]")
        return

    if args.show_table:
        print_results_table()
        return

    if args.search:
        results = search_chunks(args.search, n_results=10, section_name=args.section)
        table = Table(title=f'Search: "{args.search}"', show_lines=True)
        table.add_column("Score", width=6)
        table.add_column("Paper", width=18)
        table.add_column("Section", width=12)
        table.add_column("Text", max_width=80, overflow="fold")
        for r in results:
            table.add_row(str(r["score"]), r["paper_id"], r["section_name"], r["chunk_text"][:300])
        console.print(table)
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
