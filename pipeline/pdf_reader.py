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
    # Only drop truly empty lines or single-character noise (page numbers etc.)
    # Keep short lines — table cells like "28", "N/A", "%" must survive
    lines = [ln for ln in text.splitlines() if len(ln.strip()) > 1]
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
                # Extract tables first so they appear as structured text
                table_texts = []
                for table in page.extract_tables():
                    rows = []
                    for row in table:
                        rows.append(" | ".join(str(c or "").strip() for c in row))
                    table_texts.append("\n".join(rows))

                page_text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                combined = page_text
                if table_texts:
                    combined += "\n\n[TABLES]\n" + "\n\n".join(table_texts)
                if combined.strip():
                    pages.append(combined)
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


def get_relevant_chunk(full_text: str, max_chars: int = 80000) -> str:
    """
    Prepare paper text for LLM extraction.
    1. Strip references section (no clinical data there)
    2. Send everything else — tables, methods, results, discussion all included
    3. If still over limit (rare), trim from the tail not the middle
    """
    # Strip references — nothing useful there for extraction
    ref_match = re.search(
        r'\n(references|bibliography|literatur)\s*\n',
        full_text, re.IGNORECASE
    )
    if ref_match:
        full_text = full_text[:ref_match.start()]

    # Full paper fits — send everything (the normal case for 80k limit)
    if len(full_text) <= max_chars:
        return full_text

    # Over limit: keep the front (abstract + methods + tables) and trim the tail
    # Do NOT skip methods — Table 1 baseline characteristics lives there
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
