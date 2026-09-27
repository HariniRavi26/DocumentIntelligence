"""
summarizer.py
-------------
Document summarization using the Groq LLM.

For short documents, we summarize the text directly.
For long documents (more chunks than fit comfortably in one prompt), we use
a simple map-reduce approach:
  1. Summarize each group of chunks ("map")
  2. Summarize the combined mini-summaries into one final summary ("reduce")
This avoids blowing past the model's context window on large uploads.
"""

from __future__ import annotations
from typing import List, Dict

from langchain_groq import ChatGroq

from src.utils import GROQ_MODEL_NAME, get_groq_api_key, get_logger

logger = get_logger(__name__)

# Roughly how many characters of chunk text we bundle into one summarization
# call. Conservative, since Groq models still have finite context windows
# and we want headroom for the prompt + system instructions.
MAX_CHARS_PER_MAP_CALL = 8000


def _get_llm() -> ChatGroq:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file before "
            "requesting a summary."
        )
    return ChatGroq(model=GROQ_MODEL_NAME, api_key=api_key, temperature=0.2)


def _summarize_text(llm: ChatGroq, text: str, instruction: str) -> str:
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": "You write clear, factual summaries. Do not add "
                "information that isn't in the given text.",
            },
            {"role": "user", "content": f"{instruction}\n\n{text}"},
        ]
    )
    return response.content


def _group_chunks(chunks: List[Dict], max_chars: int) -> List[str]:
    """Bundle chunk texts into groups that stay under max_chars each."""
    groups: List[str] = []
    current: List[str] = []
    current_len = 0

    for chunk in chunks:
        text = chunk["text"]
        if current_len + len(text) > max_chars and current:
            groups.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(text)
        current_len += len(text)

    if current:
        groups.append("\n\n".join(current))
    return groups


def summarize_chunks(chunks: List[Dict]) -> str:
    """
    Produce a summary from a list of chunk dicts (as returned by
    text_processing.chunk_page_records). Uses map-reduce for large inputs.
    """
    if not chunks:
        return "No content available to summarize."

    llm = _get_llm()
    groups = _group_chunks(chunks, MAX_CHARS_PER_MAP_CALL)

    if len(groups) == 1:
        # Small enough to summarize directly
        return _summarize_text(
            llm,
            groups[0],
            "Summarize the following document in 4-6 sentences, covering the "
            "main points and any key conclusions:",
        )

    # --- Map step: summarize each group individually ---
    logger.info("Summarizing in map-reduce mode across %d groups", len(groups))
    partial_summaries = []
    for i, group_text in enumerate(groups, start=1):
        summary = _summarize_text(
            llm,
            group_text,
            f"Summarize part {i} of a larger document in 3-4 sentences, "
            "covering only its main points:",
        )
        partial_summaries.append(summary)

    # --- Reduce step: combine the partial summaries into one final summary ---
    combined = "\n\n".join(partial_summaries)
    final_summary = _summarize_text(
        llm,
        combined,
        "The following are summaries of different parts of the same "
        "document. Combine them into one coherent overall summary of "
        "6-8 sentences, covering the main findings/points:",
    )
    return final_summary
