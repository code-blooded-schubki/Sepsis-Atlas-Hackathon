"""
pipeline/verifier.py — Verifiability scoring independent of the LLM.

For every extracted value we compute:
  source_verified  — does the source_sentence the LLM cited actually exist in the PDF?
  value_verified   — do the numbers (or keywords) in the value appear in that sentence?
  verifiability_score — weighted combination, stored in DB

This is computed once at extraction time (full_text already in memory).
Query time = instant DB lookup, no re-scanning.
"""

from __future__ import annotations
import re


# ── Core checks ───────────────────────────────────────────────────────────────

def _source_in_pdf(source_sentence: str, full_text: str, window: int = 8) -> bool:
    """
    Check if the source_sentence was actually in the PDF.
    Strategy: take every consecutive window of `window` words from source_sentence
    and check if that exact string appears in full_text.
    One match is enough — we just need proof the sentence is real.
    """
    words = source_sentence.split()
    if len(words) < 3:
        # Too short to be meaningful — just check direct containment
        return source_sentence.strip().lower() in full_text.lower()

    step = min(window, len(words))
    for i in range(len(words) - step + 1):
        fragment = " ".join(words[i:i + step])
        if fragment.lower() in full_text.lower():
            return True
    return False


def _numeric_match(value: str, source_sentence: str) -> float:
    """
    What fraction of numbers in `value` appear in `source_sentence`?
    e.g. value="28.3%" source="28-day mortality was 28.3%" → 1.0
    """
    numbers = re.findall(r'\d+\.?\d*', value or "")
    if not numbers:
        return None  # signal: no numbers, fall back to keyword match
    matched = sum(1 for n in numbers if n in source_sentence)
    return matched / len(numbers)


def _keyword_overlap(value: str, source_sentence: str) -> float:
    """Fraction of meaningful words in value that appear in source_sentence."""
    stopwords = {"the", "a", "an", "of", "in", "and", "or", "was", "is",
                 "were", "are", "to", "with", "for", "that", "this", "at"}
    words = [w.lower() for w in re.findall(r'\w+', value or "") if w.lower() not in stopwords]
    if not words:
        return 0.0
    src_lower = source_sentence.lower()
    matched = sum(1 for w in words if w in src_lower)
    return matched / len(words)


def compute_verifiability(value: str, source_sentence: str, full_text: str) -> dict:
    """
    Returns {source_verified, value_verified, verifiability_score}.
    Call this at extraction time when full_text is already in memory.
    """
    if not source_sentence:
        return {"source_verified": 0, "value_verified": 0.0, "verifiability_score": 0.0}

    # Check 1: is the source sentence real?
    source_verified = int(_source_in_pdf(source_sentence, full_text))

    # Check 2: do the extracted values match the source?
    numeric = _numeric_match(value, source_sentence)
    if numeric is not None:
        value_verified = numeric
    else:
        value_verified = _keyword_overlap(value, source_sentence)

    # Weighted score: source verification counts more (hallucinated source = untrustworthy)
    score = round(0.6 * source_verified + 0.4 * value_verified, 3)

    return {
        "source_verified": source_verified,
        "value_verified": round(value_verified, 3),
        "verifiability_score": score,
    }


# ── Paper-level verifiability ─────────────────────────────────────────────────

def verify_paper(paper, full_text: str) -> dict:
    """
    Run verifiability checks on all extracted fields of a paper.
    Returns a flat dict: {field_name: verifiability_result, ...}
    and an overall_verifiability score (mean across all fields that have source sentences).
    """
    results = {}
    scores = []

    checks = [
        ("title",        paper.metadata.title),
        ("year",         paper.metadata.year),
        ("journal",      paper.metadata.journal),
        ("study_design", paper.metadata.study_design),
        ("country",      paper.metadata.country_or_region),
        ("sep_definition",      paper.sepsis_definition.definition_used),
        ("sofa",                paper.sepsis_definition.sofa_score_reported),
        ("lactate",             paper.sepsis_definition.lactate_threshold),
        ("int_primary",         paper.interventions.primary_intervention),
        ("int_antibiotics",     paper.interventions.antibiotic_protocol),
        ("out_mortality",       paper.outcomes.mortality_rate),
        ("out_icu_los",         paper.outcomes.icu_length_of_stay),
    ]

    for field_name, ef in checks:
        if ef.value and ef.source_sentence:
            v = compute_verifiability(ef.value, ef.source_sentence, full_text)
            results[field_name] = v
            scores.append(v["verifiability_score"])

    for cohort in paper.cohorts:
        if cohort.mortality_rate and cohort.source_sentence:
            v = compute_verifiability(cohort.mortality_rate, cohort.source_sentence, full_text)
            results[f"cohort_{cohort.cohort_name}_mortality"] = v
            scores.append(v["verifiability_score"])

    for i, f in enumerate(paper.prognostic_findings):
        if f.effect_size and f.source_sentence:
            v = compute_verifiability(f.effect_size, f.source_sentence, full_text)
            results[f"finding_{i}_{f.predictor}"] = v
            scores.append(v["verifiability_score"])

    overall = round(sum(scores) / len(scores), 3) if scores else 0.0
    return {"fields": results, "overall_verifiability": overall}
