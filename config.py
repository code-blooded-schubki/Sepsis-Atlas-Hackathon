"""
config.py — central configuration for the Sepsis Atlas pipeline.
Tweak these settings before the hackathon.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads from a .env file if present (recommended — don't hardcode keys)

# ── API ──────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "YOUR_KEY_HERE")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

MODEL: str = "anthropic/claude-sonnet-4.5"

MAX_TOKENS: int = 8096

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"    # drop PDFs here
OUTPUT_DIR = DATA_DIR / "outputs"   # extracted JSON/CSV land here
DB_PATH    = DATA_DIR / "sepsis_atlas.db"
CHROMA_DIR = DATA_DIR / "chroma"

# Create dirs if they don't exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ── Extraction settings ──────────────────────────────────────────────────────
CHUNK_SIZE: int = 12000

# If True, the pipeline will skip papers already in the DB (useful for re-runs)
SKIP_ALREADY_EXTRACTED: bool = True

# Minimum confidence score (0.0–1.0) to include a field in the final output
MIN_CONFIDENCE: float = 0.4

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR
