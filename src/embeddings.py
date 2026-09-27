"""
embeddings.py
-------------
Thin wrapper around a Hugging Face sentence-transformer embedding model.
Loaded once and cached, since loading the model from disk takes a few
seconds and we don't want to repeat that on every rerun.
"""

from __future__ import annotations
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils import EMBEDDING_MODEL_NAME, get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFaceEmbeddings instance.

    Model: all-MiniLM-L6-v2 — 384-dimensional embeddings, ~80MB, CPU-friendly.
    First call downloads the model from Hugging Face Hub; later calls reuse
    the local cache in ~/.cache/huggingface.
    """
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
