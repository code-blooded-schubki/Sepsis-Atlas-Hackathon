"""
pipeline/pdf_reader.py — Extract clean text from a sepsis PDF.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 5]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\xad", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
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
        import fitz
        doc = fitz.open(pdf_path)
        pages = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        logger.warning(f"PyMuPDF failed on {pdf_path.name}: {e}")
        return None
    
def extract_tables(pdf_path: Path) -> str:
    """
    Extract tables from PDF and convert to readable text format.
    Returns table content as formatted string to prepend to main text.
    """
    try:
        import pdfplumber
        table_texts = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    rows = []
                    for row in table:
                        cleaned = [str(cell).strip() if cell else "" for cell in row]
                        rows.append(" | ".join(cleaned))
                    table_text = "\n".join(rows)
                    table_texts.append(f"[TABLE page {i+1}]\n{table_text}\n[END TABLE]")
        return "\n\n".join(table_texts) if table_texts else ""
    except Exception as e:
        logger.warning(f"Table extraction failed: {e}")
        return ""


def extract_text(pdf_path: Path) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    logger.info(f"Extracting text from: {pdf_path.name}")
    text = _extract_with_pdfplumber(pdf_path)
    if not text or len(text.strip()) < 200:
        text = _extract_with_pymupdf(pdf_path)
    if not text or len(text.strip()) < 200:
        raise ValueError(f"Could not extract usable text from {pdf_path.name}")
    cleaned = _clean_text(text)
    # Extract tables and prepend — ensures they're not cut off by chunk limit
    tables_text = extract_tables(pdf_path)
    if tables_text:
        cleaned = "=== EXTRACTED TABLES ===\n" + tables_text + "\n\n=== FULL TEXT ===\n" + cleaned
        logger.info(f"Added {len(tables_text):,} chars from tables")

    logger.info(f"Extracted {len(cleaned):,} chars from {pdf_path.name}")
    return cleaned


def get_relevant_chunk(full_text: str, max_chars: int = 20000) -> str:
    """
    Return the most relevant portion of a paper for LLM extraction.
    Strategy: remove references section, then take as much as possible.
    """
    # Remove references section — not useful for extraction
    ref_match = re.search(
        r'\n(references|bibliography|literatur)\s*\n', 
        full_text, re.IGNORECASE
    )
    if ref_match:
        full_text = full_text[:ref_match.start()]

    # If short enough take everything
    if len(full_text) <= max_chars:
        return full_text

    # Find results section
    results_match = re.search(
        r'\n(results|findings|outcomes|discussion)\s*\n', 
        full_text, re.IGNORECASE
    )

    if results_match:
        # Take first 4000 chars (abstract/intro) + everything from results
        top = full_text[:4000]
        bottom = full_text[results_match.start():]
        combined = top + "\n\n--- [methods omitted] ---\n\n" + bottom
        return combined[:max_chars]
    
    return full_text[:max_chars]


def get_page_count(pdf_path: Path) -> int:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        n = doc.page_count
        doc.close()
        return n
    except Exception:
        return -1


# ── RAG / CHUNKING — disabled for now, will be re-enabled later ───────────────

# _SECTION_PATTERNS = [
#     (re.compile(r'^\s*(abstract)\s*$', re.I | re.M), 'abstract'),
#     (re.compile(r'^\s*(introduction|background)\s*$', re.I | re.M), 'introduction'),
#     (re.compile(r'^\s*(materials?\s+and\s+methods?|methods?|methodology)\s*$', re.I | re.M), 'methods'),
#     (re.compile(r'^\s*(results?|findings?)\s*$', re.I | re.M), 'results'),
#     (re.compile(r'^\s*(discussion)\s*$', re.I | re.M), 'discussion'),
#     (re.compile(r'^\s*(conclusion|conclusions?)\s*$', re.I | re.M), 'conclusion'),
#     (re.compile(r'^\s*(references?|bibliography)\s*$', re.I | re.M), 'references'),
#     (re.compile(r'^\s*(acknowledgements?|funding)\s*$', re.I | re.M), 'acknowledgements'),
# ]
# CHUNK_SIZE_CHARS  = 1600
# CHUNK_OVERLAP_CHARS = 200
# _SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

# def parse_sections(full_text: str, paper_id: str) -> list[dict]: ...
# def chunk_paper(sections: list[dict], paper_id: str) -> list[dict]: ...
