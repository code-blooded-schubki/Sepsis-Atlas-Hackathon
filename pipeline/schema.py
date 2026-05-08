"""
pipeline/schema.py — Pydantic models defining what we extract from each paper.

This is the most important file to understand.
Every field has:
  - The value itself
  - source_sentence: the exact text from the paper it came from (verifiability!)
  - confidence: how confident the LLM is (0.0 – 1.0)

Tweak the fields here to match what the hackathon judges care about.
"""

from __future__ import annotations
from typing import Optional, List
from pydantic import BaseModel, Field


# ── Reusable base: every extracted field is a (value, source, confidence) triple ──

class ExtractedField(BaseModel):
    """Wrapper that adds source tracing + confidence to any extracted value."""
    value: Optional[str] = Field(None, description="The extracted value")
    source_sentence: Optional[str] = Field(
        None,
        description="The exact sentence(s) from the paper that support this value"
    )
    confidence: float = Field(
        0.0,
        ge=0.0, le=1.0,
        description="LLM confidence in this extraction (0 = uncertain, 1 = certain)"
    )


# ── Sub-models ────────────────────────────────────────────────────────────────

class StudyMetadata(BaseModel):
    title: ExtractedField = Field(default_factory=ExtractedField)
    year: ExtractedField = Field(default_factory=ExtractedField)
    journal: ExtractedField = Field(default_factory=ExtractedField)
    study_design: ExtractedField = Field(
        default_factory=ExtractedField,
        description="e.g. RCT, prospective cohort, retrospective, meta-analysis"
    )
    country_or_region: ExtractedField = Field(default_factory=ExtractedField)


class PatientPopulation(BaseModel):
    sample_size: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Total number of patients enrolled"
    )
    mean_age: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Mean or median age of patients"
    )
    percent_male: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Percentage of male patients"
    )
    clinical_setting: ExtractedField = Field(
        default_factory=ExtractedField,
        description="e.g. ICU, emergency department, general ward, mixed"
    )
    inclusion_criteria_summary: ExtractedField = Field(default_factory=ExtractedField)


class SepsisDefinition(BaseModel):
    definition_used: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Which sepsis definition: Sepsis-1, Sepsis-2, Sepsis-3, or other"
    )
    sofa_score_reported: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Was SOFA score reported? If yes, mean/median value"
    )
    qsofa_reported: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Was qSOFA reported? If yes, mean/median value"
    )
    lactate_threshold: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Lactate threshold used (mmol/L), if any"
    )
    septic_shock_included: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Did the study include septic shock patients? yes/no/unclear"
    )


class Interventions(BaseModel):
    primary_intervention: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Main treatment or intervention studied"
    )
    comparison_group: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Control or comparison group description"
    )
    antibiotic_protocol: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Antibiotic regimen, if described"
    )
    fluid_resuscitation: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Fluid resuscitation protocol, if described"
    )
    vasopressor_use: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Vasopressor type and threshold, if described"
    )


class Outcomes(BaseModel):
    primary_outcome: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Stated primary outcome of the study"
    )
    mortality_rate: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Overall or 28/30/90-day mortality rate (%)"
    )
    mortality_timepoint: ExtractedField = Field(
        default_factory=ExtractedField,
        description="At what time point was mortality measured (28-day, ICU, hospital)"
    )
    icu_length_of_stay: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Mean/median ICU length of stay (days)"
    )
    secondary_outcomes_summary: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Brief summary of secondary outcomes reported"
    )


class Cohort(BaseModel):
    """One cohort (sub-population) within a paper."""
    cohort_name: str = Field(..., description="e.g. 'KPNC non-ICU', 'UPMC derivation', 'Overall cohort'")
    sample_size: Optional[str] = None
    mean_age: Optional[str] = None
    percent_male: Optional[str] = None
    clinical_setting: Optional[str] = None
    inclusion_criteria: Optional[str] = None
    mortality_rate: Optional[str] = None
    mortality_timepoint: Optional[str] = None
    icu_length_of_stay: Optional[str] = None
    primary_outcome: Optional[str] = None
    source_sentence: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class PrognosticFinding(BaseModel):
    """One predictor→outcome association extracted from a paper."""
    predictor: Optional[str] = None
    outcome: Optional[str] = None
    timing: Optional[str] = None
    method: Optional[str] = None
    effect_size: Optional[str] = None
    performance: Optional[str] = None
    notes: Optional[str] = None
    source_sentence: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# ── Top-level paper model ─────────────────────────────────────────────────────

class ExtractedPaper(BaseModel):
    """
    Complete extraction result for one sepsis paper.
    This is what gets saved to the database and exported to CSV.
    """
    # Internal tracking
    paper_id: str = Field(..., description="Unique ID — we use the filename")
    pdf_filename: str
    extraction_timestamp: str

    # Clinical content
    metadata: StudyMetadata = Field(default_factory=StudyMetadata)
    population: PatientPopulation = Field(default_factory=PatientPopulation)
    sepsis_definition: SepsisDefinition = Field(default_factory=SepsisDefinition)
    interventions: Interventions = Field(default_factory=Interventions)
    outcomes: Outcomes = Field(default_factory=Outcomes)

    cohorts: List[Cohort] = Field(
        default_factory=list,
        description="One entry per distinct cohort/sub-population in the paper",
    )
    prognostic_findings: List[PrognosticFinding] = Field(
        default_factory=list,
        description="Predictor→outcome associations with effect sizes (for counterfactual mortality use case)"
    )

    # Overall quality signal
    overall_confidence: float = Field(
        0.0,
        description="Mean confidence across all extracted fields"
    )
    overall_verifiability: float = Field(
        0.0,
        description="Fraction of extracted values verified against source PDF text"
    )
    extraction_notes: Optional[str] = Field(
        None,
        description="Any caveats or issues the LLM flagged during extraction"
    )

    def compute_overall_confidence(self) -> float:
        """Calculate mean confidence across all leaf ExtractedField objects."""
        scores = []
        for section in [self.metadata, self.population, self.sepsis_definition,
                        self.interventions, self.outcomes]:
            for field in section.model_fields:
                ef: ExtractedField = getattr(section, field)
                if isinstance(ef, ExtractedField):
                    scores.append(ef.confidence)
        self.overall_confidence = round(sum(scores) / len(scores), 3) if scores else 0.0
        return self.overall_confidence

    def to_flat_dict(self) -> dict:
        """
        Flatten the nested structure into a single-row dict for CSV export.
        Each key is like 'outcomes.mortality_rate.value'
        """
        flat = {
            "paper_id": self.paper_id,
            "pdf_filename": self.pdf_filename,
            "extraction_timestamp": self.extraction_timestamp,
            "overall_confidence": self.overall_confidence,
            "extraction_notes": self.extraction_notes,
        }
        for section_name in ["metadata", "population", "sepsis_definition",
                              "interventions", "outcomes"]:
            section = getattr(self, section_name)
            for field_name in section.model_fields:
                ef: ExtractedField = getattr(section, field_name)
                if isinstance(ef, ExtractedField):
                    prefix = f"{section_name}.{field_name}"
                    flat[f"{prefix}.value"] = ef.value
                    flat[f"{prefix}.source"] = ef.source_sentence
                    flat[f"{prefix}.confidence"] = ef.confidence
        return flat
