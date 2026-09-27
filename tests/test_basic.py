"""
test_basic.py
-------------
A handful of quick unit tests covering the non-LLM parts of the pipeline
(text cleaning, chunking, keyword extraction, statistics, and TXT/PDF
loading). These don't require a Groq API key.

Run with:
    pytest tests/test_basic.py -v
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.text_processing import clean_text, chunk_page_records, extract_keywords, compute_statistics
from src.document_loader import load_txt, validate_file, DocumentLoadError


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \n\n  world") == "hello world"


def test_clean_text_handles_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_chunk_page_records_produces_metadata():
    records = [{"filename": "a.txt", "page": 1, "text": "word " * 500}]
    chunks = chunk_page_records(records)
    assert len(chunks) > 1
    for c in chunks:
        assert c["filename"] == "a.txt"
        assert c["page"] == 1
        assert "chunk_index" in c


def test_extract_keywords_returns_words():
    text = "machine learning machine learning artificial intelligence data data data"
    keywords = extract_keywords(text, top_n=3)
    assert "data" in keywords


def test_compute_statistics_counts_correctly():
    records = [
        {"filename": "a.txt", "page": 1, "text": "One two three. Four five."},
    ]
    stats = compute_statistics(records)
    assert stats["documents"] == 1
    assert stats["pages"] == 1
    assert stats["words"] == 5


def test_validate_file_rejects_bad_extension():
    try:
        validate_file("notes.docx", 1000)
        assert False, "Expected DocumentLoadError"
    except DocumentLoadError:
        pass


def test_validate_file_rejects_oversized_file():
    try:
        validate_file("big.pdf", 200 * 1024 * 1024)
        assert False, "Expected DocumentLoadError"
    except DocumentLoadError:
        pass


def test_load_txt_reads_content():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world, this is a test document.")
        path = f.name
    try:
        records = load_txt(path, "test.txt")
        assert len(records) == 1
        assert "Hello world" in records[0]["text"]
        assert records[0]["page"] == 1
    finally:
        os.remove(path)


def test_load_txt_rejects_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        path = f.name
    try:
        try:
            load_txt(path, "empty.txt")
            assert False, "Expected DocumentLoadError"
        except DocumentLoadError:
            pass
    finally:
        os.remove(path)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
