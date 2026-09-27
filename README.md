# Document Intelligence System

A Document Intelligence application built using NLP, Retrieval-Augmented Generation (RAG), and Generative AI.

The system allows users to upload PDF or TXT documents and ask questions about their content using natural language. It retrieves the most relevant parts of the documents and uses a Groq-hosted LLM to generate grounded answers with source information.

## Features

- Upload PDF and TXT documents
- Extract and clean document text
- Split documents into meaningful chunks
- Generate semantic embeddings using Hugging Face
- Store and retrieve embeddings using ChromaDB
- Natural-language question answering using RAG
- Source information for generated answers
- Document summarization
- Keyword extraction
- Document statistics
- Multiple document support
- Error handling for invalid or empty documents

## NLP Used

NLP is used throughout the document processing and question-answering pipeline:

1. Text extraction and preprocessing
2. Text cleaning
3. Text chunking
4. Semantic text embeddings
5. Similarity-based semantic search
6. Keyword extraction
7. Document summarization
8. Natural-language question answering

## RAG Pipeline

PDF/TXT Document
↓
Text Extraction
↓
Text Cleaning
↓
Text Chunking
↓
Hugging Face Embeddings
↓
ChromaDB
↓
User Question
↓
Question Embedding
↓
Similarity Search
↓
Relevant Document Chunks
↓
Groq LLM
↓
Grounded Answer + Sources

## Technologies

- Python
- Streamlit
- LangChain
- PyMuPDF
- Hugging Face
- Sentence Transformers
- ChromaDB
- Groq
- NLTK

## Project Structure

DocumentIntelligence/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── sample_docs/
├── data/
├── chroma_db/
├── src/
│   ├── document_loader.py
│   ├── text_processing.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── rag_pipeline.py
│   ├── summarizer.py
│   └── utils.py
└── tests/

## How It Works

The uploaded document is first converted into text and divided into smaller chunks. Each chunk is converted into a semantic embedding and stored in ChromaDB.

When the user asks a question, the question is converted into an embedding and compared with the stored document embeddings. The most relevant chunks are retrieved and passed to the Groq LLM as context. The LLM then generates an answer based on the retrieved information.

## Project Outcome

The system combines NLP, semantic search, vector databases, RAG, and Generative AI to provide question answering and document analysis from user-provided documents.