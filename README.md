# 📄 Document Intelligence System

A Retrieval-Augmented Generation (RAG) web app that lets you upload PDF/TXT
documents and ask questions about them. Answers are grounded strictly in the
uploaded content, with source citations (filename + page + chunk) for every
answer — the app explicitly says so if the answer isn't in your documents,
instead of guessing.

## What is RAG, and why use it?

A general-purpose LLM only knows what it was trained on — it has never seen
*your* documents, and it can "hallucinate" plausible-sounding but wrong
answers when asked about them directly.

**Retrieval-Augmented Generation** fixes this by:
1. Breaking your documents into small chunks and turning each into a vector
   (an "embedding") that captures its meaning.
2. When you ask a question, embedding the question the same way and finding
   the chunks whose vectors are most similar ("similarity search").
3. Feeding only those relevant chunks to the LLM as context, and instructing
   it to answer *using only that context*.

This grounds the answer in your actual documents and lets you see exactly
which passages it came from.

## Features

- Upload multiple PDF/TXT documents
- Text extraction (page-aware for PDFs) → cleaning → chunking → embedding
- ChromaDB vector store with persistent local storage
- Top-K similarity search with an adjustable K
- Groq-hosted LLM for fast answer generation, grounded in retrieved chunks
- Source citations (filename, page, chunk index) shown per answer
- "Not found in documents" response when nothing relevant is retrieved
- Document summarization (direct or map-reduce for large documents)
- Keyword extraction and document statistics
- Clear/reset the knowledge base from the sidebar

## Architecture

```
User
 -> Streamlit UI
 -> Document Upload (PDF/TXT)
 -> Text Extraction (PyMuPDF / plain read)
 -> Text Cleaning (text_processing.py)
 -> Chunking (RecursiveCharacterTextSplitter, 1000 chars / 150 overlap)
 -> Embeddings (HuggingFace all-MiniLM-L6-v2)
 -> ChromaDB Vector Store (persistent, local)
 -> [User Question] -> Question Embedding -> Similarity Search (top-K)
 -> Relevant Chunks -> Prompt Construction -> Groq LLM
 -> Grounded Answer + Sources -> Streamlit UI
```

### Project structure

```
DocumentIntelligence/
├── app.py                  # Streamlit UI — the only file you run
├── requirements.txt
├── README.md
├── .env.example            # copy to .env and add your Groq key
├── .gitignore
├── data/uploads/           # uploaded files land here temporarily
├── chroma_db/              # persistent Chroma vector index
└── src/
    ├── document_loader.py  # PDF/TXT text extraction
    ├── text_processing.py  # cleaning, chunking, keywords, stats
    ├── embeddings.py       # HuggingFace embedding wrapper
    ├── vector_store.py     # ChromaDB wrapper (add/search/clear)
    ├── rag_pipeline.py     # retrieval + Groq LLM answer generation
    ├── summarizer.py       # map-reduce summarization
    └── utils.py            # shared config/constants
```

## Local setup (Windows / VS Code)

1. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Get a free Groq API key:** https://console.groq.com/keys

4. **Configure your key:**
   ```powershell
   copy .env.example .env
   ```
   Then open `.env` and paste your key:
   ```
   GROQ_API_KEY=gsk_your_real_key_here
   ```
   `.env` is already listed in `.gitignore`, so it will never be committed.

5. **Run the app:**
   ```powershell
   streamlit run app.py
   ```
   It should open automatically at `http://localhost:8501`.

## Key RAG concepts used in this project

| Concept | Where it's implemented |
|---|---|
| Document loading | `document_loader.py` — PyMuPDF for PDFs, plain read for TXT |
| Text splitting / chunk size / overlap | `text_processing.py` — 1000 chars, 150 overlap |
| Embeddings | `embeddings.py` — `all-MiniLM-L6-v2`, 384-dim, CPU |
| Vector database | `vector_store.py` — ChromaDB, cosine similarity |
| Similarity search / top-K retrieval | `VectorStore.similarity_search()` |
| Context construction & prompting | `rag_pipeline.py` — `_build_context`, `SYSTEM_PROMPT` |
| LLM generation | `rag_pipeline.py` — Groq `ChatGroq` |
| Source attribution | metadata (filename/page/chunk) carried through every stage |

## Error handling

The app handles, with a readable Streamlit message for each:
- Empty documents / empty pages
- Scanned/image-only PDFs (no extractable text)
- Unsupported file types
- Files over the size limit
- Missing `GROQ_API_KEY`
- Groq API errors (bad key, network issues)
- No relevant chunks retrieved → explicit "not found" answer
- Re-uploading the same file (chunks are upserted, not duplicated)

## Deployment to Streamlit Community Cloud

1. **Push to GitHub:**
   ```powershell
   git init
   git add .
   git commit -m "Initial commit: Document Intelligence System"
   git branch -M main
   git remote add origin https://github.com/<your-username>/DocumentIntelligence.git
   git push -u origin main
   ```
   (`.env` and `chroma_db/` are git-ignored, so your API key and local
   vector index are never pushed.)

2. **Deploy:** go to https://share.streamlit.io, connect your GitHub repo,
   and set the main file to `app.py`.

3. **Streamlit secrets:** in your app's dashboard, go to **Settings → Secrets**
   and add:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   ```
   `python-dotenv`'s `load_dotenv()` won't find this file on Streamlit Cloud,
   but `os.getenv("GROQ_API_KEY")` still works because Streamlit injects
   secrets as environment variables automatically — no code changes needed.

4. **ChromaDB persistence on Streamlit Cloud:** the filesystem on Streamlit
   Community Cloud is **ephemeral** — it resets whenever the app restarts,
   sleeps from inactivity, or redeploys. That means the local `chroma_db/`
   folder will **not survive** a redeploy or restart.

   For a portfolio demo this is usually fine (users just re-upload their
   documents each session). If you need persistence across restarts:
   - Simplest fix: keep local Chroma, and just accept that the knowledge
     base resets on redeploy — document this clearly for viewers.
   - If persistence is a hard requirement, use a small hosted vector DB
     with a free tier (e.g. Chroma Cloud, or Qdrant Cloud) instead of the
     local `PersistentClient` — swap only `vector_store.py`, nothing else
     changes. Only do this if you actually need it; it adds an external
     dependency for a feature most portfolio reviewers won't test.

## Resume-ready project description

> **Document Intelligence System** — Built a full-stack Retrieval-Augmented
> Generation (RAG) application in Python/Streamlit that lets users upload
> PDF/TXT documents and ask natural-language questions grounded strictly in
> their content. Implemented the complete RAG pipeline: PyMuPDF text
> extraction, chunking with LangChain, sentence-transformer embeddings,
> a persistent ChromaDB vector store, top-K similarity retrieval, and
> Groq-hosted LLM generation with per-answer source citations (file, page,
> chunk). Added map-reduce document summarization, keyword extraction, and
> document statistics, with error handling for scanned PDFs, missing API
> keys, and empty retrieval results.

## Interview explanation (30-second version)

"I built an app where you upload documents and ask questions about them, and
the answers are backed only by what's actually in those documents — not the
model's general knowledge. Under the hood it's a RAG pipeline: I split
documents into chunks, embed them with a sentence-transformer model, store
the vectors in ChromaDB, and at query time retrieve the most relevant chunks
by cosine similarity before passing them to a Groq-hosted LLM to generate the
final answer. Every answer comes with the exact source chunk it was based
on, and if nothing relevant is found, it says so instead of guessing."

## Likely interview questions & answers

**Q: What is RAG and why not just fine-tune the model on your documents?**
A: RAG retrieves relevant context at query time instead of baking documents
into model weights. It's cheaper, works with any document you add later
without retraining, and — critically — lets you cite exactly which source
an answer came from, which fine-tuning can't do.

**Q: Why did you choose this chunk size and overlap?**
A: 1000 characters keeps chunks small enough to be topically focused (so
retrieval is precise) while staying large enough to preserve context. A 150
character overlap prevents a sentence or idea from being awkwardly split
right at a chunk boundary, so meaning isn't lost between chunks.

**Q: Why cosine similarity instead of another distance metric?**
A: Cosine similarity measures the angle between vectors rather than their
magnitude, which suits normalized sentence embeddings well — it captures
semantic similarity without being skewed by text length.

**Q: How do you prevent hallucination?**
A: The system prompt instructs the LLM to answer only from the provided
context and to explicitly say when the answer isn't found there. Retrieval
also acts as a filter — if no relevant chunks are retrieved at all, the app
returns the "not found" message without even calling the LLM.

**Q: How would you scale this to thousands of documents?**
A: Move from a local `PersistentClient` to a hosted/managed vector database
(e.g. Chroma Cloud, Pinecone, Qdrant) for concurrent access and larger
indexes, batch the embedding step, and consider approximate nearest-neighbor
indexing parameters (e.g. HNSW `ef`/`M`) for faster search at scale.

**Q: What are the limitations of this approach?**
A: Retrieval quality depends entirely on chunking and embedding quality —
poor chunk boundaries or an embedding model that misses domain-specific
nuance will hurt retrieval. It also can't answer questions requiring
reasoning across information that's never co-located in any single
retrieved chunk (multi-hop reasoning), and scanned/image PDFs need OCR,
which this app doesn't include.

**Q: Why Groq instead of OpenAI/Anthropic directly?**
A: Groq runs open-weight models (like Llama) on custom LPU hardware, giving
very low-latency inference, and offers a generous free tier — a good fit
for a fast, responsive demo without incurring high API costs.

## Optional future enhancements

- DOCX support (straightforward to add via `python-docx`, not included to
  keep the initial scope focused on PDF/TXT)
- OCR for scanned/image-only PDFs
- Multi-hop / conversational memory across questions
- Hosted vector DB for persistence across Streamlit Cloud redeploys
