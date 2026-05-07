"""
utils/db.py — Storage layer.

SQLite tables:
  papers    — one row per paper, paper-level metadata
  cohorts   — one row per cohort per paper (population + outcomes)
  raw_extractions — full JSON blob per paper (for debugging)

RAG / ChromaDB / chunking is disabled for now — will be re-enabled later.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

import config
from pipeline.schema import ExtractedPaper
from utils.logger import get_logger

logger = get_logger(__name__)


def _engine():
    return create_engine(f"sqlite:///{config.DB_PATH}", echo=False)


# ── Init ───────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _engine().connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id              TEXT PRIMARY KEY,
                pdf_filename          TEXT,
                extraction_timestamp  TEXT,
                overall_confidence    REAL,
                extraction_notes      TEXT,

                meta_title            TEXT, meta_title_src TEXT, meta_title_conf REAL,
                meta_year             TEXT, meta_year_src TEXT, meta_year_conf REAL,
                meta_journal          TEXT, meta_journal_src TEXT, meta_journal_conf REAL,
                meta_study_design     TEXT, meta_study_design_src TEXT, meta_study_design_conf REAL,
                meta_country          TEXT, meta_country_src TEXT, meta_country_conf REAL,

                sep_definition        TEXT, sep_definition_src TEXT, sep_definition_conf REAL,
                sep_sofa              TEXT, sep_sofa_src TEXT, sep_sofa_conf REAL,
                sep_qsofa             TEXT, sep_qsofa_src TEXT, sep_qsofa_conf REAL,
                sep_lactate           TEXT, sep_lactate_src TEXT, sep_lactate_conf REAL,
                sep_shock             TEXT, sep_shock_src TEXT, sep_shock_conf REAL,

                int_primary           TEXT, int_primary_src TEXT, int_primary_conf REAL,
                int_comparison        TEXT, int_comparison_src TEXT, int_comparison_conf REAL,
                int_antibiotics       TEXT, int_antibiotics_src TEXT, int_antibiotics_conf REAL,
                int_fluids            TEXT, int_fluids_src TEXT, int_fluids_conf REAL,
                int_vasopressors      TEXT, int_vasopressors_src TEXT, int_vasopressors_conf REAL,

                prog_findings         TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS cohorts (
                cohort_id            TEXT PRIMARY KEY,
                paper_id             TEXT NOT NULL,
                cohort_name          TEXT,
                sample_size          TEXT,
                mean_age             TEXT,
                percent_male         TEXT,
                clinical_setting     TEXT,
                inclusion_criteria   TEXT,
                mortality_rate       TEXT,
                mortality_timepoint  TEXT,
                icu_length_of_stay   TEXT,
                primary_outcome      TEXT,
                source_sentence      TEXT,
                confidence           REAL,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            )
        """))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_cohorts_paper ON cohorts(paper_id)"
        ))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_extractions (
                paper_id             TEXT PRIMARY KEY,
                raw_json             TEXT,
                extraction_timestamp TEXT
            )
        """))
        conn.commit()
    logger.info(f"Database initialised at {config.DB_PATH}")


# ── Papers ─────────────────────────────────────────────────────────────────────

def paper_exists(paper_id: str) -> bool:
    with _engine().connect() as conn:
        return conn.execute(
            text("SELECT 1 FROM papers WHERE paper_id = :pid"), {"pid": paper_id}
        ).fetchone() is not None


def save_paper(paper: ExtractedPaper) -> None:
    m, s, i = paper.metadata, paper.sepsis_definition, paper.interventions

    row = {
        "paper_id": paper.paper_id,
        "pdf_filename": paper.pdf_filename,
        "extraction_timestamp": paper.extraction_timestamp,
        "overall_confidence": paper.overall_confidence,
        "extraction_notes": paper.extraction_notes,

        "meta_title": m.title.value, "meta_title_src": m.title.source_sentence, "meta_title_conf": m.title.confidence,
        "meta_year": m.year.value, "meta_year_src": m.year.source_sentence, "meta_year_conf": m.year.confidence,
        "meta_journal": m.journal.value, "meta_journal_src": m.journal.source_sentence, "meta_journal_conf": m.journal.confidence,
        "meta_study_design": m.study_design.value, "meta_study_design_src": m.study_design.source_sentence, "meta_study_design_conf": m.study_design.confidence,
        "meta_country": m.country_or_region.value, "meta_country_src": m.country_or_region.source_sentence, "meta_country_conf": m.country_or_region.confidence,

        "sep_definition": s.definition_used.value, "sep_definition_src": s.definition_used.source_sentence, "sep_definition_conf": s.definition_used.confidence,
        "sep_sofa": s.sofa_score_reported.value, "sep_sofa_src": s.sofa_score_reported.source_sentence, "sep_sofa_conf": s.sofa_score_reported.confidence,
        "sep_qsofa": s.qsofa_reported.value, "sep_qsofa_src": s.qsofa_reported.source_sentence, "sep_qsofa_conf": s.qsofa_reported.confidence,
        "sep_lactate": s.lactate_threshold.value, "sep_lactate_src": s.lactate_threshold.source_sentence, "sep_lactate_conf": s.lactate_threshold.confidence,
        "sep_shock": s.septic_shock_included.value, "sep_shock_src": s.septic_shock_included.source_sentence, "sep_shock_conf": s.septic_shock_included.confidence,

        "int_primary": i.primary_intervention.value, "int_primary_src": i.primary_intervention.source_sentence, "int_primary_conf": i.primary_intervention.confidence,
        "int_comparison": i.comparison_group.value, "int_comparison_src": i.comparison_group.source_sentence, "int_comparison_conf": i.comparison_group.confidence,
        "int_antibiotics": i.antibiotic_protocol.value, "int_antibiotics_src": i.antibiotic_protocol.source_sentence, "int_antibiotics_conf": i.antibiotic_protocol.confidence,
        "int_fluids": i.fluid_resuscitation.value, "int_fluids_src": i.fluid_resuscitation.source_sentence, "int_fluids_conf": i.fluid_resuscitation.confidence,
        "int_vasopressors": i.vasopressor_use.value, "int_vasopressors_src": i.vasopressor_use.source_sentence, "int_vasopressors_conf": i.vasopressor_use.confidence,

        "prog_findings": json.dumps([f.model_dump() for f in paper.prognostic_findings]),
    }

    with _engine().connect() as conn:
        conn.execute(text("DELETE FROM papers WHERE paper_id = :pid"), {"pid": paper.paper_id})
        conn.execute(text("DELETE FROM raw_extractions WHERE paper_id = :pid"), {"pid": paper.paper_id})
        placeholders = ", ".join(f":{k}" for k in row)
        cols = ", ".join(row.keys())
        conn.execute(text(f"INSERT INTO papers ({cols}) VALUES ({placeholders})"), row)
        conn.execute(text(
            "INSERT INTO raw_extractions (paper_id, raw_json, extraction_timestamp) "
            "VALUES (:pid, :rj, :ts)"
        ), {"pid": paper.paper_id, "rj": paper.model_dump_json(), "ts": paper.extraction_timestamp})
        conn.commit()
    logger.debug(f"Saved paper {paper.paper_id}")


# ── Cohorts ────────────────────────────────────────────────────────────────────

def save_cohorts(paper: ExtractedPaper) -> None:
    if not paper.cohorts:
        return
    with _engine().connect() as conn:
        conn.execute(text("DELETE FROM cohorts WHERE paper_id = :pid"), {"pid": paper.paper_id})
        for idx, c in enumerate(paper.cohorts):
            cohort_id = f"{paper.paper_id}_c{idx:02d}"
            conn.execute(text("""
                INSERT INTO cohorts (
                    cohort_id, paper_id, cohort_name, sample_size, mean_age,
                    percent_male, clinical_setting, inclusion_criteria,
                    mortality_rate, mortality_timepoint, icu_length_of_stay,
                    primary_outcome, source_sentence, confidence
                ) VALUES (
                    :cohort_id, :paper_id, :cohort_name, :sample_size, :mean_age,
                    :percent_male, :clinical_setting, :inclusion_criteria,
                    :mortality_rate, :mortality_timepoint, :icu_length_of_stay,
                    :primary_outcome, :source_sentence, :confidence
                )
            """), {
                "cohort_id": cohort_id,
                "paper_id": paper.paper_id,
                "cohort_name": c.cohort_name,
                "sample_size": c.sample_size,
                "mean_age": c.mean_age,
                "percent_male": c.percent_male,
                "clinical_setting": c.clinical_setting,
                "inclusion_criteria": c.inclusion_criteria,
                "mortality_rate": c.mortality_rate,
                "mortality_timepoint": c.mortality_timepoint,
                "icu_length_of_stay": c.icu_length_of_stay,
                "primary_outcome": c.primary_outcome,
                "source_sentence": c.source_sentence,
                "confidence": c.confidence,
            })
        conn.commit()
    logger.debug(f"Saved {len(paper.cohorts)} cohort(s) for {paper.paper_id}")


# ── Load / export ──────────────────────────────────────────────────────────────

def load_all_papers() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM papers", _engine())


def load_cohorts(paper_id: Optional[str] = None) -> pd.DataFrame:
    if paper_id:
        return pd.read_sql(
            "SELECT * FROM cohorts WHERE paper_id = :pid",
            _engine(), params={"pid": paper_id},
        )
    return pd.read_sql("SELECT * FROM cohorts", _engine())


def load_cohorts_with_paper_meta() -> pd.DataFrame:
    """Cohorts joined with paper-level metadata — the main analysis table."""
    return pd.read_sql("""
        SELECT
            c.cohort_id, c.paper_id, c.cohort_name,
            p.meta_title   AS title,
            p.meta_year    AS year,
            p.meta_journal AS journal,
            p.meta_study_design AS study_design,
            p.meta_country AS country,
            p.sep_definition AS sepsis_definition,
            c.sample_size, c.mean_age, c.percent_male,
            c.clinical_setting, c.inclusion_criteria,
            c.mortality_rate, c.mortality_timepoint,
            c.icu_length_of_stay, c.primary_outcome,
            c.confidence
        FROM cohorts c
        JOIN papers p ON c.paper_id = p.paper_id
        ORDER BY p.meta_year DESC, c.paper_id, c.cohort_id
    """, _engine())


def export_to_csv(output_path: Optional[Path] = None) -> Path:
    output_path = output_path or config.OUTPUT_DIR / "sepsis_atlas_extracted.csv"
    df = load_cohorts_with_paper_meta()
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} cohort rows to {output_path}")
    return output_path


# ── RAG / ChromaDB — disabled for now, will be re-enabled later ───────────────

# def _chroma(): ...
# def _chunk_collection(): ...
# def save_sections(sections): ...
# def save_chunks(chunks): ...
# def search_chunks(query, n_results, section_name): ...
