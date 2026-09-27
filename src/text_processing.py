"""
text_processing.py
-------------------
Basic NLP utilities:
  - clean_text(): strip weird whitespace/artifacts from extracted text
  - chunk_page_records(): split page records into overlapping chunks with metadata
  - extract_keywords(): simple frequency-based keyword extraction (NLTK)
  - compute_statistics(): word/char/sentence counts etc.
"""

from __future__ import annotations
import re
import string
from collections import Counter
from typing import List, Dict

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import CHUNK_SIZE, CHUNK_OVERLAP, get_logger

logger = get_logger(__name__)

# Make sure the NLTK data we need is present. Downloads silently once,
# then reuses the local cache on every later run.
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(
            f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}"
        )
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass  # we degrade gracefully below if data truly isn't available

from nltk.corpus import stopwords  # noqa: E402
from nltk.tokenize import word_tokenize, sent_tokenize  # noqa: E402

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = set()


def clean_text(text: str) -> str:
    """Normalize whitespace and drop non-printable junk from extracted text."""
    if not text:
        return ""
    # Collapse repeated whitespace/newlines
    text = re.sub(r"\s+", " ", text)
    # Remove characters that aren't printable (common in messy PDF extracts)
    text = "".join(ch for ch in text if ch in string.printable or ch.isalpha())
    return text.strip()


def chunk_page_records(page_records: List[Dict]) -> List[Dict]:
    """
    Split each page's text into overlapping chunks, preserving metadata.

    Input:  [{"filename": "a.pdf", "page": 1, "text": "..."}, ...]
    Output: [{"filename": "a.pdf", "page": 1, "chunk_index": 0, "text": "..."}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict] = []
    for record in page_records:
        cleaned = clean_text(record["text"])
        if not cleaned:
            continue
        pieces = splitter.split_text(cleaned)
        for idx, piece in enumerate(pieces):
            chunks.append(
                {
                    "filename": record["filename"],
                    "page": record["page"],
                    "chunk_index": idx,
                    "text": piece,
                }
            )
    return chunks


def extract_keywords(text: str, top_n: int = 15) -> List[str]:
    """
    Very lightweight keyword extraction: tokenize, drop stopwords/punctuation,
    and return the most frequent remaining words. Good enough for a portfolio
    project without pulling in a heavy model.
    """
    if not text:
        return []
    try:
        words = word_tokenize(text.lower())
    except LookupError:
        words = text.lower().split()

    words = [
        w for w in words
        if w.isalpha() and w not in STOPWORDS and len(w) > 2
    ]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(top_n)]


def compute_statistics(page_records: List[Dict]) -> Dict:
    """Return simple document statistics used in the UI's stats panel."""
    full_text = " ".join(r["text"] for r in page_records)
    try:
        sentences = sent_tokenize(full_text)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", full_text)

    words = full_text.split()
    unique_files = {r["filename"] for r in page_records}
    total_pages = len({(r["filename"], r["page"]) for r in page_records})

    return {
        "documents": len(unique_files),
        "pages": total_pages,
        "characters": len(full_text),
        "words": len(words),
        "sentences": len(sentences),
    }
