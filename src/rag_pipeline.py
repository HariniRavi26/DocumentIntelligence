"""
rag_pipeline.py
----------------
Ties retrieval (vector_store.py) together with generation (Groq LLM) to
answer a user's question using only the retrieved document chunks.

Flow:
  question -> similarity_search() -> build context -> prompt Groq LLM
  -> grounded answer + the sources used to produce it
"""

from __future__ import annotations
from typing import List, Dict

from langchain_groq import ChatGroq

from src.vector_store import VectorStore
from src.utils import GROQ_MODEL_NAME, get_groq_api_key, DEFAULT_TOP_K, get_logger

logger = get_logger(__name__)

NOT_FOUND_PHRASE = "I couldn't find this information in the uploaded documents."

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using "
    "ONLY the context excerpts provided below, which come from documents the "
    "user uploaded. Follow these rules strictly:\n"
    "1. Base your answer only on the given context. Do not use outside knowledge.\n"
    "2. If the context does not contain enough information to answer, respond "
    f'exactly with: "{NOT_FOUND_PHRASE}" — do not guess or make anything up.\n'
    "3. Be concise and directly answer the question.\n"
    "4. When helpful, you may briefly mention which document a fact came from."
)


class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self._llm = None  # lazily created once an API key is confirmed present

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            api_key = get_groq_api_key()
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your .env file "
                    "(see .env.example) before asking questions."
                )
            self._llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                api_key=api_key,
                temperature=0.1,
            )
        return self._llm

    @staticmethod
    def _build_context(chunks: List[Dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, start=1):
            parts.append(
                f"[Source {i}: {c['filename']}, page {c['page']}]\n{c['text']}"
            )
        return "\n\n".join(parts)

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> Dict:
        """
        Returns:
            {
                "answer": str,
                "sources": [ {filename, page, chunk_index, text}, ... ],
                "found_context": bool
            }
        """
        if not question or not question.strip():
            return {"answer": "Please enter a question.", "sources": [], "found_context": False}

        chunks = self.vector_store.similarity_search(question, top_k=top_k)

        if not chunks:
            return {
                "answer": NOT_FOUND_PHRASE,
                "sources": [],
                "found_context": False,
            }

        context = self._build_context(chunks)
        user_message = (
            f"Context excerpts from uploaded documents:\n\n{context}\n\n"
            f"Question: {question}"
        )

        try:
            llm = self._get_llm()
            response = llm.invoke(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ]
            )
            answer_text = response.content
        except RuntimeError as e:
            # Missing API key — surface directly to the UI
            raise
        except Exception as e:
            logger.error("Groq API call failed: %s", e)
            raise RuntimeError(
                "The LLM request failed. This is usually an invalid/expired "
                f"Groq API key or a network issue. Details: {e}"
            )

        return {
            "answer": answer_text,
            "sources": chunks,
            "found_context": True,
        }
