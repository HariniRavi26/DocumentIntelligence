"""
document_loader.py
-------------------
Turns an uploaded file (PDF or TXT) into a list of "page records":
    {"filename": ..., "page": ..., "text": ...}

For PDFs, one record is produced per page (so we can cite page numbers
later). For TXT files, there is only one "page" (page = 1).
"""

from __future__ import annotations
import os
from typing import List, Dict

import fitz  # PyMuPDF

from src.utils import get_logger, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

logger = get_logger(__name__)


class DocumentLoadError(Exception):
    """Raised when a file can't be read or is unsupported."""


def validate_file(filename: str, size_bytes: int) -> None:
    """Raise DocumentLoadError with a clear message if the file is invalid."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{ext}'. Only PDF and TXT are supported."
        )
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise DocumentLoadError(
            f"'{filename}' is {size_mb:.1f} MB, which exceeds the "
            f"{MAX_FILE_SIZE_MB} MB limit for this demo app."
        )


def load_pdf(file_path: str, filename: str) -> List[Dict]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    records: List[Dict] = []
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise DocumentLoadError(f"Could not open '{filename}' as a PDF: {e}")

    if doc.page_count == 0:
        doc.close()
        raise DocumentLoadError(f"'{filename}' has no pages.")

    empty_pages = 0
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        if not text:
            empty_pages += 1
            continue
        records.append({"filename": filename, "page": page_num + 1, "text": text})
    doc.close()

    if not records:
        # Every page came back empty -> almost certainly a scanned/image PDF
        raise DocumentLoadError(
            f"'{filename}' appears to contain no extractable text "
            "(it may be a scanned/image-only PDF). OCR is not enabled in "
            "this app, so please upload a text-based PDF."
        )
    if empty_pages:
        logger.info("%s: skipped %d empty page(s)", filename, empty_pages)
    return records


def load_txt(file_path: str, filename: str) -> List[Dict]:
    """Read a plain text file, trying a couple of common encodings."""
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read().strip()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise DocumentLoadError(f"Could not decode '{filename}' as text.")

    if not text:
        raise DocumentLoadError(f"'{filename}' is empty.")

    return [{"filename": filename, "page": 1, "text": text}]


def load_document(file_path: str, filename: str) -> List[Dict]:
    """Dispatch to the right loader based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path, filename)
    elif ext == ".txt":
        return load_txt(file_path, filename)
    else:
        raise DocumentLoadError(f"Unsupported file type '{ext}'.")
