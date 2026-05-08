"""
test_pipeline.py — Test that the full pipeline works correctly.

Tests:
1. PDF text extraction
2. LLM extraction (requires API key)
3. Database correctly populated
4. All required fields present
5. Prognostic findings extracted

Usage:
    python test_pipeline.py --file data/papers/Besen_2016.pdf
    python test_pipeline.py --all   # test all PDFs in data/papers/
"""

import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


# ── Test 1: PDF extraction ─────────────────────────────────────────────────

def test_pdf_extraction(pdf_path: Path) -> str:
    console.print(f"\n[bold]Test 1: PDF extraction[/bold] — {pdf_path.name}")
    
    from pipeline.pdf_reader import extract_text, get_relevant_chunk
    
    text = extract_text(pdf_path)
    assert len(text) > 200, "Extracted text too short"
    console.print(f"  [green]✓[/green] Extracted {len(text):,} characters")
    
    chunk = get_relevant_chunk(text, max_chars=12000)
    assert len(chunk) > 100, "Chunk too short"
    console.print(f"  [green]✓[/green] Chunk: {len(chunk):,} characters")
    
    return chunk


# ── Test 2: LLM extraction ─────────────────────────────────────────────────

def test_llm_extraction(chunk: str, pdf_filename: str):
    console.print(f"\n[bold]Test 2: LLM extraction[/bold]")
    
    from pipeline.extractor import extract_paper
    
    paper = extract_paper(chunk, pdf_filename)
    
    # Check metadata
    assert paper.metadata.title.value, "Title not extracted"
    console.print(f"  [green]✓[/green] Title: {paper.metadata.title.value[:60]}")
    
    assert paper.metadata.year.value, "Year not extracted"
    console.print(f"  [green]✓[/green] Year: {paper.metadata.year.value}")
    
    # Check population
    assert paper.population.sample_size.value, "Sample size not extracted"
    console.print(f"  [green]✓[/green] Sample size: {paper.population.sample_size.value}")
    
    # Check outcomes
    assert paper.outcomes.mortality_rate.value, "Mortality rate not extracted"
    console.print(f"  [green]✓[/green] Mortality: {paper.outcomes.mortality_rate.value}")
    
    # Check prognostic findings
    n_findings = len(paper.prognostic_findings)
    console.print(f"  [green]✓[/green] Prognostic findings: {n_findings}")
    if n_findings == 0:
        console.print(f"  [yellow]⚠[/yellow] No prognostic findings — check prompt")
    
    # Check cohorts
    n_cohorts = len(paper.cohorts)
    console.print(f"  [green]✓[/green] Cohorts: {n_cohorts}")
    
    # Check confidence
    paper.compute_overall_confidence()
    console.print(f"  [green]✓[/green] Overall confidence: {paper.overall_confidence:.0%}")
    if paper.overall_confidence < 0.5:
        console.print(f"  [yellow]⚠[/yellow] Low confidence — paper may be problematic")
    
    return paper


# ── Test 3: Database ───────────────────────────────────────────────────────

def test_database(paper):
    console.print(f"\n[bold]Test 3: Database[/bold]")
    
    from utils.db import init_db, save_paper, save_cohorts, save_findings, paper_exists
    from utils.db import load_all_papers, load_cohorts
    import pandas as pd
    
    init_db()
    save_paper(paper)
    save_cohorts(paper)
    save_findings(paper)
    
    # Check paper saved
    assert paper_exists(paper.paper_id), "Paper not saved"
    console.print(f"  [green]✓[/green] Paper saved: {paper.paper_id}")
    
    # Check papers table
    df_papers = load_all_papers()
    row = df_papers[df_papers["paper_id"] == paper.paper_id]
    assert not row.empty, "Paper not in papers table"
    console.print(f"  [green]✓[/green] Papers table: {len(df_papers)} total rows")
    
    # Check cohorts table
    df_cohorts = load_cohorts(paper.paper_id)
    console.print(f"  [green]✓[/green] Cohorts table: {len(df_cohorts)} cohort(s) for this paper")
    
    # Check findings table
    from sqlalchemy import create_engine, text
    import config
    engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)
    with engine.connect() as conn:
        n_findings = conn.execute(
            text("SELECT COUNT(*) FROM findings WHERE paper_id = :pid"),
            {"pid": paper.paper_id}
        ).scalar()
    console.print(f"  [green]✓[/green] Findings table: {n_findings} finding(s) for this paper")
    
    if n_findings == 0:
        console.print(f"  [yellow]⚠[/yellow] No findings saved — check save_findings()")


# ── Test 4: Data quality ───────────────────────────────────────────────────

def test_data_quality(paper):
    console.print(f"\n[bold]Test 4: Data quality[/bold]")
    
    issues = 0
    
    # Check source sentences
    fields_to_check = [
        ("mortality_rate", paper.outcomes.mortality_rate),
        ("sample_size", paper.population.sample_size),
        ("sepsis_definition", paper.sepsis_definition.definition_used),
    ]
    
    for name, ef in fields_to_check:
        if ef.value and not ef.source_sentence:
            console.print(f"  [yellow]⚠[/yellow] {name} has value but no source_sentence")
            issues += 1
        elif ef.value and ef.source_sentence:
            console.print(f"  [green]✓[/green] {name}: has value + source")
        else:
            console.print(f"  [yellow]⚠[/yellow] {name}: not reported")
    
    # Check findings quality
    for i, f in enumerate(paper.prognostic_findings):
        if not f.source_sentence:
            console.print(f"  [yellow]⚠[/yellow] Finding {i+1} ({f.predictor}) has no source_sentence")
            issues += 1
        if not f.effect_size:
            console.print(f"  [yellow]⚠[/yellow] Finding {i+1} ({f.predictor}) has no effect_size")
    
    if issues == 0:
        console.print(f"  [green]✓[/green] All quality checks passed")
    else:
        console.print(f"  [yellow]⚠[/yellow] {issues} quality issue(s) found")


# ── Summary table ──────────────────────────────────────────────────────────

def print_summary(paper):
    console.print(f"\n[bold]Summary[/bold]")
    
    table = Table(show_lines=True)
    table.add_column("Field", width=25)
    table.add_column("Value", width=50)
    table.add_column("Confidence", width=12)
    
    rows = [
        ("Title", paper.metadata.title),
        ("Year", paper.metadata.year),
        ("Study design", paper.metadata.study_design),
        ("Country", paper.metadata.country_or_region),
        ("Sample size", paper.population.sample_size),
        ("Clinical setting", paper.population.clinical_setting),
        ("Sepsis definition", paper.sepsis_definition.definition_used),
        ("Mortality rate", paper.outcomes.mortality_rate),
        ("Mortality timepoint", paper.outcomes.mortality_timepoint),
    ]
    
    for label, ef in rows:
        val = (ef.value or "not reported")[:50]
        conf = f"{ef.confidence:.0%}" if ef.value else "—"
        table.add_row(label, val, conf)
    
    console.print(table)
    
    if paper.prognostic_findings:
        console.print(f"\n[bold]Prognostic findings ({len(paper.prognostic_findings)})[/bold]")
        f_table = Table(show_lines=True)
        f_table.add_column("Predictor", width=25)
        f_table.add_column("Outcome", width=20)
        f_table.add_column("Effect size", width=30)
        f_table.add_column("Conf.", width=8)
        
        for f in paper.prognostic_findings:
            f_table.add_row(
                (f.predictor or "—")[:25],
                (f.outcome or "—")[:20],
                (f.effect_size or "—")[:30],
                f"{f.confidence:.0%}"
            )
        console.print(f_table)


# ── Main ───────────────────────────────────────────────────────────────────

def run_test(pdf_path: Path):
    console.print(f"\n{'='*60}")
    console.print(f"[bold cyan]Testing: {pdf_path.name}[/bold cyan]")
    console.print(f"{'='*60}")
    
    try:
        chunk = test_pdf_extraction(pdf_path)
        paper = test_llm_extraction(chunk, pdf_path.name)
        test_database(paper)
        test_data_quality(paper)
        print_summary(paper)
        console.print(f"\n[bold green]✓ All tests passed for {pdf_path.name}[/bold green]\n")
        return True
    except AssertionError as e:
        console.print(f"\n[bold red]✗ Test failed: {e}[/bold red]\n")
        return False
    except Exception as e:
        console.print(f"\n[bold red]✗ Unexpected error: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Test a single PDF")
    parser.add_argument("--all", action="store_true", help="Test all PDFs")
    args = parser.parse_args()
    
    import config
    
    if args.file:
        pdf_path = Path(args.file)
        run_test(pdf_path)
    
    elif args.all:
        pdf_paths = sorted(config.PAPERS_DIR.glob("*.pdf"))
        if not pdf_paths:
            console.print(f"[yellow]No PDFs in {config.PAPERS_DIR}[/yellow]")
            return
        
        results = []
        for pdf_path in pdf_paths:
            ok = run_test(pdf_path)
            results.append((pdf_path.name, ok))
        
        console.print(f"\n[bold]Final Results[/bold]")
        for name, ok in results:
            status = "[green]✓[/green]" if ok else "[red]✗[/red]"
            console.print(f"  {status} {name}")
        
        passed = sum(1 for _, ok in results if ok)
        console.print(f"\n{passed}/{len(results)} passed")
    
    else:
        console.print("Use --file <path> or --all")


if __name__ == "__main__":
    main()