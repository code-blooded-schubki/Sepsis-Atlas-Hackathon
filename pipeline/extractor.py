"""
pipeline/extractor.py — Send paper text to Claude, get back structured data.

This is the heart of the pipeline.
The LLM receives the paper text + a detailed prompt, and must return
a JSON object matching our ExtractedPaper schema.
"""

from __future__ import annotations
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

import config
from pipeline.schema import (
    ExtractedPaper, ExtractedField,
    StudyMetadata, PatientPopulation, SepsisDefinition,
    Interventions, Outcomes, PrognosticFinding,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.OPENROUTER_BASE_URL)


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical data extraction specialist. Your job is to extract
structured information from sepsis research papers and return it as valid JSON.

Rules:
1. For EVERY field, include the exact sentence(s) from the paper that support your answer
   in "source_sentence". This is mandatory — it allows verification.
2. Set "confidence" between 0.0 and 1.0:
   - 0.9–1.0: value is stated explicitly and clearly
   - 0.6–0.8: value is implied or requires minor inference
   - 0.3–0.5: value is unclear, ambiguous, or inferred from context
   - 0.0–0.2: value not found or not reported
3. If a field is not reported in the paper, set value to null and confidence to 0.0.
4. Use clinical terminology correctly (Sepsis-3 = Singer 2016 definition, SOFA ≥ 2, etc.)
5. Return ONLY valid JSON — no prose, no markdown code fences, no comments.
"""

def _build_extraction_prompt(paper_text: str) -> str:
    return f"""Extract structured information from the following sepsis research paper.

Return a JSON object with exactly this structure. For every field, provide:
- "value": the extracted string (or null if not found)
- "source_sentence": the exact sentence(s) from the paper supporting this value
- "confidence": float 0.0–1.0

JSON structure to fill:
{{
  "metadata": {{
    "title": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "year": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "journal": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "study_design": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "country_or_region": {{"value": null, "source_sentence": null, "confidence": 0.0}}
  }},
  "population": {{
    "sample_size": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "mean_age": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "percent_male": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "clinical_setting": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "inclusion_criteria_summary": {{"value": null, "source_sentence": null, "confidence": 0.0}}
  }},
  "sepsis_definition": {{
    "definition_used": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "sofa_score_reported": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "qsofa_reported": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "lactate_threshold": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "septic_shock_included": {{"value": null, "source_sentence": null, "confidence": 0.0}}
  }},
  "interventions": {{
    "primary_intervention": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "comparison_group": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "antibiotic_protocol": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "fluid_resuscitation": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "vasopressor_use": {{"value": null, "source_sentence": null, "confidence": 0.0}}
  }},
  "outcomes": {{
    "primary_outcome": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "mortality_rate": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "mortality_timepoint": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "icu_length_of_stay": {{"value": null, "source_sentence": null, "confidence": 0.0}},
    "secondary_outcomes_summary": {{"value": null, "source_sentence": null, "confidence": 0.0}}
  }},
  "prognostic_findings": [
    {{
      "predictor": null,
      "outcome": null,
      "timing": null,
      "method": null,
      "effect_size": null,
      "performance": null,
      "notes": null,
      "source_sentence": null,
      "confidence": 0.0
    }}
  ],
  "extraction_notes": null
}}

prognostic_findings is an array — include one object per predictor→outcome association reported.
Common predictors: lactate, IL-6, lymphocytes, SOFA, APACHE II, procalcitonin, CRP, age, comorbidities.
effect_size: AUC, OR (with CI), HR (with CI), regression coefficient, cutoff value, etc.
performance: sensitivity, specificity, PPV, NPV if reported.

--- PAPER TEXT START ---
{paper_text}
--- PAPER TEXT END ---

Return only the filled JSON object."""


# ── Core extraction function ──────────────────────────────────────────────────

def extract_paper(paper_text: str, pdf_filename: str) -> ExtractedPaper:
    """
    Send paper text to Claude and parse the structured JSON response.

    Args:
        paper_text: cleaned text from the PDF (use pdf_reader.get_relevant_chunk)
        pdf_filename: used as the paper_id and for logging

    Returns:
        ExtractedPaper with all fields populated
    """
    logger.info(f"Sending to LLM: {pdf_filename} ({len(paper_text):,} chars)")

    response = _client.chat.completions.create(
        model=config.MODEL,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_extraction_prompt(paper_text)},
        ],
    )

    raw_text = response.choices[0].message.content
    logger.debug(f"Raw LLM response ({len(raw_text)} chars): {raw_text[:300]}...")

    # Parse the JSON
    extracted_json = _parse_json_response(raw_text)

    # Build the ExtractedPaper object from the JSON
    paper = _json_to_extracted_paper(extracted_json, pdf_filename)
    paper.compute_overall_confidence()

    logger.info(
        f"Extracted {pdf_filename} — overall confidence: {paper.overall_confidence:.2f}"
    )
    return paper


def _parse_json_response(raw: str) -> dict:
    """
    Robustly parse the LLM JSON response.
    Handles cases where the model wraps output in ```json ... ``` fences.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}\nRaw response:\n{raw[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


def _ef(data: dict, key: str) -> ExtractedField:
    """Helper: pull an ExtractedField from a nested dict, with safe defaults."""
    node = data.get(key, {})
    if not isinstance(node, dict):
        return ExtractedField()
    return ExtractedField(
        value=node.get("value"),
        source_sentence=node.get("source_sentence"),
        confidence=float(node.get("confidence", 0.0)),
    )


def _parse_findings(raw: list) -> list:
    findings = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        findings.append(PrognosticFinding(
            predictor=item.get("predictor"),
            outcome=item.get("outcome"),
            timing=item.get("timing"),
            method=item.get("method"),
            effect_size=item.get("effect_size"),
            performance=item.get("performance"),
            notes=item.get("notes"),
            source_sentence=item.get("source_sentence"),
            confidence=float(item.get("confidence", 0.0)),
        ))
    return findings


def _json_to_extracted_paper(data: dict, pdf_filename: str) -> ExtractedPaper:
    """Map the raw JSON dict onto our Pydantic models."""
    m = data.get("metadata", {})
    p = data.get("population", {})
    s = data.get("sepsis_definition", {})
    i = data.get("interventions", {})
    o = data.get("outcomes", {})

    return ExtractedPaper(
        paper_id=Path(pdf_filename).stem,
        pdf_filename=pdf_filename,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),

        metadata=StudyMetadata(
            title=_ef(m, "title"),
            year=_ef(m, "year"),
            journal=_ef(m, "journal"),
            study_design=_ef(m, "study_design"),
            country_or_region=_ef(m, "country_or_region"),
        ),
        population=PatientPopulation(
            sample_size=_ef(p, "sample_size"),
            mean_age=_ef(p, "mean_age"),
            percent_male=_ef(p, "percent_male"),
            clinical_setting=_ef(p, "clinical_setting"),
            inclusion_criteria_summary=_ef(p, "inclusion_criteria_summary"),
        ),
        sepsis_definition=SepsisDefinition(
            definition_used=_ef(s, "definition_used"),
            sofa_score_reported=_ef(s, "sofa_score_reported"),
            qsofa_reported=_ef(s, "qsofa_reported"),
            lactate_threshold=_ef(s, "lactate_threshold"),
            septic_shock_included=_ef(s, "septic_shock_included"),
        ),
        interventions=Interventions(
            primary_intervention=_ef(i, "primary_intervention"),
            comparison_group=_ef(i, "comparison_group"),
            antibiotic_protocol=_ef(i, "antibiotic_protocol"),
            fluid_resuscitation=_ef(i, "fluid_resuscitation"),
            vasopressor_use=_ef(i, "vasopressor_use"),
        ),
        outcomes=Outcomes(
            primary_outcome=_ef(o, "primary_outcome"),
            mortality_rate=_ef(o, "mortality_rate"),
            mortality_timepoint=_ef(o, "mortality_timepoint"),
            icu_length_of_stay=_ef(o, "icu_length_of_stay"),
            secondary_outcomes_summary=_ef(o, "secondary_outcomes_summary"),
        ),
        extraction_notes=data.get("extraction_notes"),
        prognostic_findings=_parse_findings(data.get("prognostic_findings", [])),
    )
