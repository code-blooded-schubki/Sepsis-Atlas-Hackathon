"""
backfill_verifiability.py — Compute verifiability for already-extracted papers.
No API calls needed — reads PDFs + stored source sentences from DB.

Usage: python backfill_verifiability.py
"""

import json
from pathlib import Path
from sqlalchemy import create_engine, text
from rich.console import Console
from tqdm import tqdm

import config
from pipeline.pdf_reader import extract_text
from pipeline.verifier import compute_verifiability

console = Console()
engine = create_engine(f"sqlite:///{config.DB_PATH}", echo=False)


def backfill():
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT p.paper_id, p.pdf_filename, r.raw_json "
            "FROM papers p JOIN raw_extractions r ON p.paper_id = r.paper_id "
            "WHERE p.overall_verifiability IS NULL"
        )).fetchall()

    if not rows:
        console.print("[green]All papers already have verifiability scores.[/green]")
        return

    console.print(f"Backfilling verifiability for {len(rows)} papers...\n")

    for paper_id, pdf_filename, raw_json in tqdm(rows, unit="paper"):
        pdf_path = config.PAPERS_DIR / pdf_filename
        if not pdf_path.exists():
            console.print(f"[yellow]PDF not found: {pdf_filename}, skipping[/yellow]")
            continue

        try:
            full_text = extract_text(pdf_path)
            data = json.loads(raw_json)
        except Exception as e:
            console.print(f"[red]Error loading {pdf_filename}: {e}[/red]")
            continue

        scores = []

        # Check all top-level ExtractedField values
        for section in ["metadata", "population", "sepsis_definition", "interventions", "outcomes"]:
            section_data = data.get(section, {})
            for field_name, field in section_data.items():
                if not isinstance(field, dict):
                    continue
                value = field.get("value")
                source = field.get("source_sentence")
                if value and source:
                    v = compute_verifiability(str(value), source, full_text)
                    scores.append(v["verifiability_score"])

        # Check cohorts
        for cohort in data.get("cohorts", []):
            if cohort.get("mortality_rate") and cohort.get("source_sentence"):
                v = compute_verifiability(
                    str(cohort["mortality_rate"]), cohort["source_sentence"], full_text
                )
                scores.append(v["verifiability_score"])

        # Check prognostic findings
        for finding in data.get("prognostic_findings", []):
            if finding.get("effect_size") and finding.get("source_sentence"):
                v = compute_verifiability(
                    str(finding["effect_size"]), finding["source_sentence"], full_text
                )
                scores.append(v["verifiability_score"])

        overall = round(sum(scores) / len(scores), 3) if scores else 0.0

        with engine.connect() as conn:
            conn.execute(text(
                "UPDATE papers SET overall_verifiability = :v WHERE paper_id = :pid"
            ), {"v": overall, "pid": paper_id})
            conn.commit()

    console.print("\n[green]Done. Verifiability scores updated.[/green]")

    # Show results
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT paper_id, overall_confidence, overall_verifiability FROM papers ORDER BY overall_verifiability DESC"
        )).fetchall()

    from rich.table import Table
    t = Table(title="Verifiability Scores", show_lines=True)
    t.add_column("Paper", max_width=30)
    t.add_column("Confidence")
    t.add_column("Verifiability")
    for paper_id, conf, ver in rows:
        conf_str = f"{conf:.2f}" if conf is not None else "N/A"
        ver_str  = f"{ver:.2f}"  if ver  is not None else "N/A"
        t.add_row(paper_id, conf_str, ver_str)
    console.print(t)


if __name__ == "__main__":
    backfill()
