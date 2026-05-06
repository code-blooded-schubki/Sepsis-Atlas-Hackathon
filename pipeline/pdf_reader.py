"""
pipeline/pdf_reader.py — Extract clean text from a sepsis PDF.

Tries pdfplumber first (better for text-heavy papers), falls back to PyMuPDF.
Returns the full text plus a per-section breakdown when possible.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Remove junk characters and normalise whitespace."""
    # Remove headers/footers heuristic: lines < 6 chars are usually page numbers
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 5]
    text = "\n".join(lines)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove soft hyphens and ligatures
    text = text.replace("\xad", "").replace("\ufb01", "fi").replace("\ufb02", "fl")
    return text.strip()


def _extract_with_pdfplumber(pdf_path: Path) -> Optional[str]:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    pages.append(page_text)
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        logger.warning(f"pdfplumber failed on {pdf_path.name}: {e}")
        return None


def _extract_with_pymupdf(pdf_path: Path) -> Optional[str]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        logger.warning(f"PyMuPDF failed on {pdf_path.name}: {e}")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def extract_text(pdf_path: Path) -> str:
    """
    Extract and clean text from a PDF.
    Returns cleaned full text, or raises ValueError if both extractors fail.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Extracting text from: {pdf_path.name}")

    text = _extract_with_pdfplumber(pdf_path)
    if not text or len(text.strip()) < 200:
        logger.info(f"pdfplumber returned little text, trying PyMuPDF...")
        text = _extract_with_pymupdf(pdf_path)

    if not text or len(text.strip()) < 200:
        raise ValueError(f"Could not extract usable text from {pdf_path.name}")

    cleaned = _clean_text(text)
    logger.info(f"Extracted {len(cleaned):,} characters from {pdf_path.name}")
    return cleaned


def get_relevant_chunk(full_text: str, max_chars: int = 8000) -> str:
    """
    Return the most relevant portion of a paper for extraction.

    Strategy: take the abstract + intro (top) and the results + discussion (bottom).
    Clinical papers frontload the key numbers in abstract and results sections.

    If the paper is short enough, return everything.
    """
    if len(full_text) <= max_chars:
        return full_text

    # Try to find section headers to get abstract + results
    abstract_match = re.search(
        r"(abstract|introduction|background)", full_text[:3000], re.IGNORECASE
    )
    results_match = re.search(
        r"(results|findings|outcomes|discussion)", full_text, re.IGNORECASE
    )

    if results_match:
        # Take first 3000 chars (abstract/intro) + from results onwards
        top = full_text[:3000]
        bottom_start = results_match.start()
        bottom = full_text[bottom_start: bottom_start + (max_chars - 3000)]
        chunk = top + "\n\n--- [middle sections omitted] ---\n\n" + bottom
    else:
        # No section headers found — just take the first max_chars
        chunk = full_text[:max_chars]

    return chunk


def get_page_count(pdf_path: Path) -> int:
    """Return number of pages in a PDF."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return -1
