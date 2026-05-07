"""
pipeline/pdf_reader.py — Extract text, parse sections, and chunk for embeddings.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Section detection ─────────────────────────────────────────────────────────

_SECTION_PATTERNS = [
    (re.compile(r'^\s*(abstract)\s*$', re.I | re.M), 'abstract'),
    (re.compile(r'^\s*(introduction|background)\s*$', re.I | re.M), 'introduction'),
    (re.compile(r'^\s*(materials?\s+and\s+methods?|methods?|methodology)\s*$', re.I | re.M), 'methods'),
    (re.compile(r'^\s*(results?|findings?)\s*$', re.I | re.M), 'results'),
    (re.compile(r'^\s*(discussion)\s*$', re.I | re.M), 'discussion'),
    (re.compile(r'^\s*(conclusion|conclusions?)\s*$', re.I | re.M), 'conclusion'),
    (re.compile(r'^\s*(references?|bibliography)\s*$', re.I | re.M), 'references'),
    (re.compile(r'^\s*(acknowledgements?|funding)\s*$', re.I | re.M), 'acknowledgements'),
]

CHUNK_SIZE_CHARS  = 1600  # ~400 tokens
CHUNK_OVERLAP_CHARS = 200  # ~50 tokens
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


# ── Text extraction ───────────────────────────────────────────────────────────

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
    logger.info(f"Extracted {len(cleaned):,} chars from {pdf_path.name}")
    return cleaned


# ── Section parsing ───────────────────────────────────────────────────────────

def parse_sections(full_text: str, paper_id: str) -> list[dict]:
    """
    Split full paper text into named sections.
    Returns list of {section_id, paper_id, section_name, section_text}.
    Falls back to a single 'body' section if no headers detected.
    """
    # Find all section header positions
    hits = []
    for pattern, name in _SECTION_PATTERNS:
        for m in pattern.finditer(full_text):
            hits.append((m.start(), m.end(), name))

    hits.sort(key=lambda x: x[0])

    if not hits:
        return [{
            "section_id": f"{paper_id}_body",
            "paper_id": paper_id,
            "section_name": "body",
            "section_text": full_text.strip(),
        }]

    sections = []
    name_counts: dict[str, int] = {}
    for i, (__, end, name) in enumerate(hits):
        next_start = hits[i + 1][0] if i + 1 < len(hits) else len(full_text)
        text = full_text[end:next_start].strip()
        if len(text) < 30:
            continue
        count = name_counts.get(name, 0)
        section_id = f"{paper_id}_{name}" if count == 0 else f"{paper_id}_{name}_{count}"
        name_counts[name] = count + 1
        sections.append({
            "section_id": section_id,
            "paper_id": paper_id,
            "section_name": name,
            "section_text": text,
        })

    # Capture any text before the first detected section as 'preamble'
    if hits[0][0] > 200:
        preamble = full_text[:hits[0][0]].strip()
        sections.insert(0, {
            "section_id": f"{paper_id}_preamble",
            "paper_id": paper_id,
            "section_name": "preamble",
            "section_text": preamble,
        })

    return sections


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_paper(sections: list[dict], paper_id: str) -> list[dict]:
    """
    Chunk each section into ~400-token pieces with overlap.
    Returns list of {chunk_id, paper_id, section_id, section_name, chunk_text}.
    Chunks never cross section boundaries.
    """
    chunks = []
    chunk_num = 0

    for sec in sections:
        if sec["section_name"] == "references":
            continue  # skip references — noisy for embeddings

        sentences = _SENTENCE_END.split(sec["section_text"])
        buffer = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(buffer) + len(sentence) > CHUNK_SIZE_CHARS:
                if len(buffer.strip()) >= 50:
                    chunks.append({
                        "chunk_id": f"{paper_id}_{chunk_num:04d}",
                        "paper_id": paper_id,
                        "section_id": sec["section_id"],
                        "section_name": sec["section_name"],
                        "chunk_text": buffer.strip(),
                    })
                    chunk_num += 1
                buffer = buffer[-CHUNK_OVERLAP_CHARS:] + " " + sentence
            else:
                buffer += (" " if buffer else "") + sentence

        if len(buffer.strip()) >= 50:
            chunks.append({
                "chunk_id": f"{paper_id}_{chunk_num:04d}",
                "paper_id": paper_id,
                "section_id": sec["section_id"],
                "section_name": sec["section_name"],
                "chunk_text": buffer.strip(),
            })
            chunk_num += 1

    return chunks


# ── Legacy helper (used by extractor.py for LLM call) ────────────────────────

def get_relevant_chunk(full_text: str, max_chars: int = 8000) -> str:
    if len(full_text) <= max_chars:
        return full_text

    results_match = re.search(
        r"(results|findings|outcomes|discussion)", full_text, re.IGNORECASE
    )
    if results_match:
        top = full_text[:3000]
        bottom_start = results_match.start()
        bottom = full_text[bottom_start: bottom_start + (max_chars - 3000)]
        return top + "\n\n--- [middle sections omitted] ---\n\n" + bottom

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
