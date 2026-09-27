"""
vector_store.py
----------------
Wraps a persistent ChromaDB collection. Handles:
  - adding chunked text with metadata (filename, page, chunk_index)
  - similarity search (top-k retrieval)
  - clearing the whole knowledge base
  - listing which filenames are currently stored
"""

from __future__ import annotations
from typing import List, Dict

import chromadb
from chromadb.config import Settings

from src.embeddings import get_embedding_function
from src.utils import CHROMA_DIR, CHROMA_COLLECTION_NAME, DEFAULT_TOP_K, get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedding_function()

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Embed and store a list of chunk dicts:
        {"filename", "page", "chunk_index", "text"}.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        ids = [
            f"{c['filename']}::p{c['page']}::c{c['chunk_index']}" for c in chunks
        ]
        metadatas = [
            {
                "filename": c["filename"],
                "page": c["page"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]

        embeddings = self._embedder.embed_documents(texts)

        # Upsert so re-uploading the same file safely overwrites old chunks
        # instead of creating duplicates.
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    def similarity_search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """Return the top_k most similar chunks to the query, with metadata."""
        if self._collection.count() == 0:
            return []

        query_embedding = self._embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
        )

        hits: List[Dict] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "text": text,
                    "filename": meta.get("filename"),
                    "page": meta.get("page"),
                    "chunk_index": meta.get("chunk_index"),
                    "distance": dist,
                }
            )
        return hits

    def list_filenames(self) -> List[str]:
        """Return the distinct filenames currently stored in the collection."""
        if self._collection.count() == 0:
            return []
        data = self._collection.get(include=["metadatas"])
        filenames = {m["filename"] for m in data["metadatas"] if m}
        return sorted(filenames)

    def clear(self) -> None:
        """Delete the entire collection and recreate it empty."""
        self._client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()
