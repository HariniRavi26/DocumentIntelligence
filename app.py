"""
app.py
------
Streamlit entry point for the Document Intelligence System.

Run with:  streamlit run app.py
"""

import os
import streamlit as st

from src.utils import ensure_dirs, UPLOAD_DIR, DEFAULT_TOP_K, get_groq_api_key, get_logger
from src.document_loader import load_document, validate_file, DocumentLoadError
from src.text_processing import chunk_page_records, extract_keywords, compute_statistics
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.summarizer import summarize_chunks

logger = get_logger(__name__)

st.set_page_config(page_title="Document Intelligence System", page_icon="📄", layout="wide")
ensure_dirs()


# ---------------------------------------------------------------------------
# Cached resources (created once per session, reused across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource
def get_pipeline(_vs: VectorStore) -> RAGPipeline:
    return RAGPipeline(_vs)


vector_store = get_vector_store()
pipeline = get_pipeline(vector_store)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "all_page_records" not in st.session_state:
    st.session_state.all_page_records = []  # accumulated across uploads this session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {question, answer, sources}
if "summary" not in st.session_state:
    st.session_state.summary = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Documents")

    if not get_groq_api_key():
        st.warning(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "key before asking questions or generating summaries.",
            icon="⚠️",
        )

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process uploaded files", type="primary"):
        with st.spinner("Extracting text, chunking, and embedding..."):
            total_chunks_added = 0
            for uf in uploaded_files:
                try:
                    file_bytes = uf.getvalue()
                    validate_file(uf.name, len(file_bytes))

                    save_path = os.path.join(UPLOAD_DIR, uf.name)
                    with open(save_path, "wb") as f:
                        f.write(file_bytes)

                    page_records = load_document(save_path, uf.name)
                    st.session_state.all_page_records.extend(page_records)

                    chunks = chunk_page_records(page_records)
                    added = vector_store.add_chunks(chunks)
                    total_chunks_added += added

                    st.success(f"✅ {uf.name}: {len(page_records)} page(s), {added} chunk(s) indexed.")
                except DocumentLoadError as e:
                    st.error(f"❌ {uf.name}: {e}")
                except Exception as e:
                    logger.error("Unexpected error processing %s: %s", uf.name, e)
                    st.error(f"❌ {uf.name}: unexpected error — {e}")

            st.session_state.summary = None  # invalidate old summary
            if total_chunks_added:
                st.info(f"Total: {total_chunks_added} new chunk(s) added to the knowledge base.")

    st.divider()

    stored_files = vector_store.list_filenames()
    st.subheader(f"Knowledge base ({len(stored_files)} file(s))")
    if stored_files:
        for fname in stored_files:
            st.text(f"• {fname}")
    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    st.divider()
    if st.button("🗑️ Clear knowledge base", type="secondary"):
        vector_store.clear()
        st.session_state.all_page_records = []
        st.session_state.chat_history = []
        st.session_state.summary = None
        st.success("Knowledge base cleared.")
        st.rerun()

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("📄 Document Intelligence System")
st.caption(
    "Upload PDF or TXT documents, then ask questions grounded strictly in "
    "their content — with sources cited for every answer."
)

has_documents = vector_store.count() > 0

col_stats, col_summary = st.columns([1, 2])

with col_stats:
    st.subheader("📊 Document statistics")
    if st.session_state.all_page_records:
        stats = compute_statistics(st.session_state.all_page_records)
        st.metric("Documents", stats["documents"])
        st.metric("Pages", stats["pages"])
        st.metric("Words", f"{stats['words']:,}")
        st.metric("Sentences", f"{stats['sentences']:,}")
    else:
        st.caption("Upload a document to see statistics here.")

with col_summary:
    st.subheader("📝 Summary & keywords")
    if not has_documents:
        st.caption("Upload a document to generate a summary and keywords.")
    else:
        if st.button("Generate summary"):
            if not get_groq_api_key():
                st.error("Set GROQ_API_KEY in your .env file first.")
            else:
                with st.spinner("Summarizing (this may take a moment for large documents)..."):
                    try:
                        all_chunks = chunk_page_records(st.session_state.all_page_records)
                        st.session_state.summary = summarize_chunks(all_chunks)
                    except Exception as e:
                        st.error(f"Summarization failed: {e}")

        if st.session_state.summary:
            st.write(st.session_state.summary)

        full_text = " ".join(r["text"] for r in st.session_state.all_page_records)
        keywords = extract_keywords(full_text, top_n=12)
        if keywords:
            st.write("**Keywords:** " + ", ".join(keywords))

st.divider()

# ---------------------------------------------------------------------------
# Q&A section
# ---------------------------------------------------------------------------
st.subheader("💬 Ask a question")

if not has_documents:
    st.info("Upload and process at least one document to start asking questions.")
else:
    question = st.text_input("Your question", placeholder="e.g. What are the main findings?")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        if not get_groq_api_key():
            st.error("Set GROQ_API_KEY in your .env file first.")
        else:
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                try:
                    result = pipeline.answer(question, top_k=top_k)
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                except RuntimeError as e:
                    st.error(str(e))

    # Show chat history, most recent first
    for entry in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown(f"**Q: {entry['question']}**")
            st.write(entry["answer"])
            if entry["sources"]:
                with st.expander(f"📎 Sources ({len(entry['sources'])})"):
                    for i, src in enumerate(entry["sources"], start=1):
                        st.markdown(
                            f"**Source {i}** — `{src['filename']}`, page {src['page']}, "
                            f"chunk {src['chunk_index']}"
                        )
                        st.caption(src["text"][:400] + ("..." if len(src["text"]) > 400 else ""))
