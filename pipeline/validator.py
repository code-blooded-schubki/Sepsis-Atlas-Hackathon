"""
pipeline/validator.py — Validate and quality-check extracted papers.

Provides:
- Field-level confidence filtering
- Consistency checks (e.g. mortality can't be > 100%)
- A human-readable summary of extraction quality
"""

from __future__ import annotations
from typing import List, Tuple
from pipeline.schema import ExtractedPaper, ExtractedField
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Validation rules ──────────────────────────────────────────────────────────

def validate_paper(paper: ExtractedPaper) -> Tuple[bool, List[str]]:
    """
    Run validation checks on an extracted paper.

    Returns:
        (is_valid, list_of_warnings)
        is_valid = False means we should flag this paper for manual review
    """
    warnings: List[str] = []

    # 1. Check overall confidence
    if paper.overall_confidence < 0.3:
        warnings.append(
            f"Low overall confidence ({paper.overall_confidence:.2f}) — "
            "paper may be in poor format or not a clinical study"
        )

    # 2. Core fields must be present for a useful extraction
    critical_fields = [
        ("metadata.title", paper.metadata.title),
        ("population.sample_size", paper.population.sample_size),
        ("outcomes.mortality_rate", paper.outcomes.mortality_rate),
    ]
    for field_name, ef in critical_fields:
        if ef.value is None or ef.confidence < 0.3:
            warnings.append(f"Critical field missing or low-confidence: {field_name}")

    # 3. Numeric sanity checks
    mortality = paper.outcomes.mortality_rate
    if mortality.value is not None:
        # Try to parse a number out of the mortality string
        import re
        nums = re.findall(r"\d+\.?\d*", str(mortality.value))
        if nums:
            m_val = float(nums[0])
            if m_val > 100:
                warnings.append(
                    f"Mortality rate {m_val} > 100% — likely parsing error"
                )
            if m_val > 80:
                warnings.append(
                    f"Unusually high mortality ({m_val}%) — please verify"
                )

    # 4. Source sentence check — warn if key fields lack source tracing
    fields_needing_sources = [
        paper.outcomes.mortality_rate,
        paper.population.sample_size,
        paper.sepsis_definition.definition_used,
    ]
    missing_sources = sum(
        1 for ef in fields_needing_sources
        if ef.value is not None and ef.source_sentence is None
    )
    if missing_sources > 0:
        warnings.append(
            f"{missing_sources} key field(s) have values but no source sentence"
        )

    is_valid = len([w for w in warnings if "Critical" in w or "Low overall" in w]) == 0
    return is_valid, warnings


def filter_low_confidence_fields(
    paper: ExtractedPaper, min_confidence: float = 0.4
) -> ExtractedPaper:
    """
    Set fields below min_confidence to null to keep the output clean.
    Modifies the paper in place and returns it.
    """
    sections = [
        paper.metadata, paper.population, paper.sepsis_definition,
        paper.interventions, paper.outcomes
    ]
    nulled = 0
    for section in sections:
        for field_name in section.model_fields:
            ef: ExtractedField = getattr(section, field_name)
            if isinstance(ef, ExtractedField) and ef.confidence < min_confidence:
                ef.value = None
                ef.source_sentence = None
                nulled += 1

    if nulled > 0:
        logger.debug(f"Nulled {nulled} fields below confidence {min_confidence}")
    return paper


def summarise_extraction(paper: ExtractedPaper) -> str:
    """
    Return a human-readable summary of the extraction for logging / demo display.
    """
    lines = [
        f"📄 {paper.pdf_filename}",
        f"   Title:       {paper.metadata.title.value or 'N/A'}",
        f"   Year:        {paper.metadata.year.value or 'N/A'}",
        f"   Study type:  {paper.metadata.study_design.value or 'N/A'}",
        f"   N patients:  {paper.population.sample_size.value or 'N/A'}",
        f"   Sepsis def:  {paper.sepsis_definition.definition_used.value or 'N/A'}",
        f"   Mortality:   {paper.outcomes.mortality_rate.value or 'N/A'} "
        f"({paper.outcomes.mortality_timepoint.value or 'timepoint N/A'})",
        f"   Confidence:  {paper.overall_confidence:.0%}",
        f"   Findings:    {len(paper.prognostic_findings)} predictor→outcome association(s)",
    ]
    return "\n".join(lines)
