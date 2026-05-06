"""
utils/db.py — Save and load extracted papers to/from SQLite.

Uses SQLAlchemy Core (not ORM) to keep it simple.
Two tables:
  - papers: one row per paper, flat columns for every extracted field
  - raw_extractions: stores the full JSON blob for each paper (useful for debugging)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sqlalchemy import (
    create_engine, text, Table, Column, MetaData,
    String, Float, Text, DateTime
)
from sqlalchemy.exc import IntegrityError

import config
from pipeline.schema import ExtractedPaper
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Engine ─────────────────────────────────────────────────────────────────────

def _get_engine():
    return create_engine(f"sqlite:///{config.DB_PATH}", echo=False)


# ── Table creation ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create database tables if they don't exist."""
    engine = _get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                pdf_filename TEXT,
                extraction_timestamp TEXT,
                overall_confidence REAL,
                extraction_notes TEXT,

                -- metadata
                meta_title TEXT, meta_title_src TEXT, meta_title_conf REAL,
                meta_year TEXT, meta_year_src TEXT, meta_year_conf REAL,
                meta_journal TEXT, meta_journal_src TEXT, meta_journal_conf REAL,
                meta_study_design TEXT, meta_study_design_src TEXT, meta_study_design_conf REAL,
                meta_country TEXT, meta_country_src TEXT, meta_country_conf REAL,

                -- population
                pop_sample_size TEXT, pop_sample_size_src TEXT, pop_sample_size_conf REAL,
                pop_mean_age TEXT, pop_mean_age_src TEXT, pop_mean_age_conf REAL,
                pop_percent_male TEXT, pop_percent_male_src TEXT, pop_percent_male_conf REAL,
                pop_clinical_setting TEXT, pop_clinical_setting_src TEXT, pop_clinical_setting_conf REAL,
                pop_inclusion_criteria TEXT, pop_inclusion_criteria_src TEXT, pop_inclusion_criteria_conf REAL,

                -- sepsis definition
                sep_definition TEXT, sep_definition_src TEXT, sep_definition_conf REAL,
                sep_sofa TEXT, sep_sofa_src TEXT, sep_sofa_conf REAL,
                sep_qsofa TEXT, sep_qsofa_src TEXT, sep_qsofa_conf REAL,
                sep_lactate TEXT, sep_lactate_src TEXT, sep_lactate_conf REAL,
                sep_shock TEXT, sep_shock_src TEXT, sep_shock_conf REAL,

                -- interventions
                int_primary TEXT, int_primary_src TEXT, int_primary_conf REAL,
                int_comparison TEXT, int_comparison_src TEXT, int_comparison_conf REAL,
                int_antibiotics TEXT, int_antibiotics_src TEXT, int_antibiotics_conf REAL,
                int_fluids TEXT, int_fluids_src TEXT, int_fluids_conf REAL,
                int_vasopressors TEXT, int_vasopressors_src TEXT, int_vasopressors_conf REAL,

                -- outcomes
                out_primary TEXT, out_primary_src TEXT, out_primary_conf REAL,
                out_mortality TEXT, out_mortality_src TEXT, out_mortality_conf REAL,
                out_mortality_tp TEXT, out_mortality_tp_src TEXT, out_mortality_tp_conf REAL,
                out_icu_los TEXT, out_icu_los_src TEXT, out_icu_los_conf REAL,
                out_secondary TEXT, out_secondary_src TEXT, out_secondary_conf REAL,

                prog_findings TEXT
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_extractions (
                paper_id TEXT PRIMARY KEY,
                raw_json TEXT,
                extraction_timestamp TEXT
            )
        """))
        conn.commit()
    logger.info(f"Database initialised at {config.DB_PATH}")


def paper_exists(paper_id: str) -> bool:
    """Return True if this paper_id is already in the database."""
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM papers WHERE paper_id = :pid"), {"pid": paper_id}
        ).fetchone()
    return result is not None


def save_paper(paper: ExtractedPaper) -> None:
    """
    Save an ExtractedPaper to the database.
    Overwrites existing rows with the same paper_id.
    """
    engine = _get_engine()
    m = paper.metadata
    p = paper.population
    s = paper.sepsis_definition
    i = paper.interventions
    o = paper.outcomes

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

        "pop_sample_size": p.sample_size.value, "pop_sample_size_src": p.sample_size.source_sentence, "pop_sample_size_conf": p.sample_size.confidence,
        "pop_mean_age": p.mean_age.value, "pop_mean_age_src": p.mean_age.source_sentence, "pop_mean_age_conf": p.mean_age.confidence,
        "pop_percent_male": p.percent_male.value, "pop_percent_male_src": p.percent_male.source_sentence, "pop_percent_male_conf": p.percent_male.confidence,
        "pop_clinical_setting": p.clinical_setting.value, "pop_clinical_setting_src": p.clinical_setting.source_sentence, "pop_clinical_setting_conf": p.clinical_setting.confidence,
        "pop_inclusion_criteria": p.inclusion_criteria_summary.value, "pop_inclusion_criteria_src": p.inclusion_criteria_summary.source_sentence, "pop_inclusion_criteria_conf": p.inclusion_criteria_summary.confidence,

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

        "out_primary": o.primary_outcome.value, "out_primary_src": o.primary_outcome.source_sentence, "out_primary_conf": o.primary_outcome.confidence,
        "out_mortality": o.mortality_rate.value, "out_mortality_src": o.mortality_rate.source_sentence, "out_mortality_conf": o.mortality_rate.confidence,
        "out_mortality_tp": o.mortality_timepoint.value, "out_mortality_tp_src": o.mortality_timepoint.source_sentence, "out_mortality_tp_conf": o.mortality_timepoint.confidence,
        "out_icu_los": o.icu_length_of_stay.value, "out_icu_los_src": o.icu_length_of_stay.source_sentence, "out_icu_los_conf": o.icu_length_of_stay.confidence,
        "out_secondary": o.secondary_outcomes_summary.value, "out_secondary_src": o.secondary_outcomes_summary.source_sentence, "out_secondary_conf": o.secondary_outcomes_summary.confidence,

        "prog_findings": json.dumps([f.model_dump() for f in paper.prognostic_findings]),
    }

    raw_json = paper.model_dump_json()

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM papers WHERE paper_id = :pid"), {"pid": paper.paper_id})
        conn.execute(text("DELETE FROM raw_extractions WHERE paper_id = :pid"), {"pid": paper.paper_id})

        placeholders = ", ".join(f":{k}" for k in row)
        cols = ", ".join(row.keys())
        conn.execute(text(f"INSERT INTO papers ({cols}) VALUES ({placeholders})"), row)
        conn.execute(text(
            "INSERT INTO raw_extractions (paper_id, raw_json, extraction_timestamp) VALUES (:pid, :rj, :ts)"
        ), {"pid": paper.paper_id, "rj": raw_json, "ts": paper.extraction_timestamp})
        conn.commit()

    logger.debug(f"Saved {paper.paper_id} to database")


def load_all_papers() -> pd.DataFrame:
    """Load all extracted papers as a pandas DataFrame."""
    engine = _get_engine()
    return pd.read_sql("SELECT * FROM papers", engine)


def export_to_csv(output_path: Optional[Path] = None) -> Path:
    """Export all papers to a CSV file. Returns the path."""
    output_path = output_path or config.OUTPUT_DIR / "sepsis_atlas_extracted.csv"
    df = load_all_papers()
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} papers to {output_path}")
    return output_path
