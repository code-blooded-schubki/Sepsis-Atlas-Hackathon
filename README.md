# Sepsis Atlas

> An AI pipeline that transforms published sepsis research into a structured, queryable evidence base — with built-in source traceability and independent verifiability scoring.

---

## Directory structure

```
Sepsis-Atlas-Hackathon/
├── data/
│   ├── papers/              ← drop PDF papers here
│   └── outputs/             ← extracted JSON + CSV land here
├── pipeline/
│   ├── extractor.py         ← LLM extraction logic + prompt
│   ├── pdf_reader.py        ← PDF → text + structured tables
│   ├── schema.py            ← Pydantic models (ExtractedPaper, Cohort, etc.)
│   ├── validator.py         ← confidence filtering + extraction summary
│   └── verifier.py          ← verifiability scoring (no LLM, checks PDF directly)
├── utils/
│   ├── db.py                ← SQLite (papers, cohorts, findings) storage
│   └── logger.py            ← logging setup
├── demo/
│   └── app.py               ← Streamlit dashboard
├── run_pipeline.py          ← main entry point
├── backfill_verifiability.py ← compute verifiability for already-extracted papers
├── view_db.py               ← CLI tool to inspect the database
├── config.py                ← API keys, model, paths
└── requirements.txt
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenRouter API key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# 4. Drop PDFs into data/papers/

# 5. Run the pipeline
python run_pipeline.py

# 6. Launch the Streamlit dashboard
streamlit run demo/app.py
```

---

## What gets extracted per paper

| Category | Fields |
|---|---|
| Metadata | title, year, journal, study design, country |
| Sepsis definition | Sepsis-1/2/3, SOFA, qSOFA, lactate threshold, septic shock |
| Interventions | primary intervention, antibiotics, fluids, vasopressors |
| Cohorts | one row per sub-population — name, sample size, age, mortality, ICU LOS |
| Prognostic findings | predictor → outcome, effect size (AUC/OR/HR), method, performance |

Every field includes:
- `source_sentence` — exact quote from the paper
- `confidence` — LLM self-grading (0.0–1.0)
- `verifiability` — independent check against raw PDF (no LLM)

---

## Pipeline commands

```bash
# Process all PDFs
python run_pipeline.py

# Process a single paper
python run_pipeline.py --file data/papers/Besen_2016.pdf

# Export cohorts to CSV
python run_pipeline.py --export-csv

# Print results table in terminal
python run_pipeline.py --show-table

# Backfill verifiability scores for already-extracted papers (no API needed)
python backfill_verifiability.py
```

## Inspect the database

```bash
# Summary — paper / cohort / chunk counts
python view_db.py

# All extracted papers
python view_db.py --papers

# Sections for a specific paper
python view_db.py --sections --paper Besen_2016
```

---

## Trust scores explained

**Confidence** — the LLM grades itself when extracting each field:
- `0.9–1.0` value is explicitly stated in the paper
- `0.6–0.8` value is implied or requires minor inference
- `0.3–0.5` value is ambiguous or unclear
- `0.0–0.2` value not found / not reported

**Verifiability** — computed independently, no LLM involved:
- Checks whether the cited `source_sentence` actually exists in the raw PDF text
- Checks whether extracted numbers appear in that sentence
- Score of `0.0` means the source sentence could not be found — possible hallucination

> Confidence reflects extraction certainty. Verifiability is an independent, deterministic check against the source document.

---

## Database schema

```
papers   — one row per paper (metadata, sepsis definition, interventions)
cohorts  — one row per cohort per paper (population, mortality, outcomes)
findings — one row per predictor→outcome association (effect sizes, AUC, OR, HR)
```

All three tables join on `paper_id`. The Evidence Query page converts natural language to SQL and queries these tables directly.
