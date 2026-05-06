# Sepsis Atlas — Hackathon Pipeline

Turn unstructured clinical sepsis PDFs into structured, verifiable, analysis-ready data.

## Directory structure

```
sepsis_atlas/
├── data/
│   ├── papers/          ← drop your PDF papers here
│   └── outputs/         ← extracted JSON + CSV land here
├── pipeline/
│   ├── extractor.py     ← core LLM extraction logic
│   ├── pdf_reader.py    ← PDF → clean text
│   ├── schema.py        ← Pydantic data models (your extraction schema)
│   └── validator.py     ← confidence scoring + source tracing
├── utils/
│   ├── db.py            ← SQLite storage helpers
│   └── logger.py        ← simple logging setup
├── demo/
│   └── app.py           ← Streamlit demo app
├── run_pipeline.py      ← main entry point — run this
├── config.py            ← API keys, model settings, paths
└── requirements.txt
```

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
#    Open config.py and set ANTHROPIC_API_KEY (or use a .env file)

# 4. Drop PDFs into data/papers/

# 5. Run the pipeline
python run_pipeline.py

# 6. (Optional) Launch the demo
streamlit run demo/app.py
```

## What gets extracted

See `pipeline/schema.py` for the full list. Key fields per paper:
- Study metadata (title, year, journal, design)
- Patient population (sample size, age, setting)
- Sepsis definition used (Sepsis-1/2/3, qSOFA, SOFA score)
- Key interventions
- Primary outcomes (mortality, ICU stay, etc.)
- Source sentence for every extracted field (verifiability!)
- Confidence score per field

## Hackathon tips

- Source tracing is your killer feature — every field stores the exact sentence it came from
- Run on 20+ papers to impress judges with scale
- The demo shows input PDF side-by-side with extracted structured output
