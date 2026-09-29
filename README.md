
# 📄 Document Intelligence System

A Retrieval-Augmented Generation (RAG), Natural Language Processing (NLP) application built with Python and Streamlit that allows users to upload PDF and TXT documents, ask questions, generate summaries, extract keywords, and view document statistics.

The application retrieves relevant information from uploaded documents and generates answers using a Groq-hosted LLM with source references.

## 🚀 Live Demo

🔗 **[Try Document Intelligence](https://documentintelligence-hsfrpxxh6yz7jfbskerzrh.streamlit.app/)**

> Upload a PDF or TXT document and ask questions based on its content.

## ✨ Features

- Upload PDF and TXT documents
- Extract and process document text
- Split documents into smaller chunks
- Generate embeddings using Hugging Face
- Store and retrieve document chunks using ChromaDB
- Ask questions using Retrieval-Augmented Generation (RAG)
- Generate answers using Groq LLM
- Display source references (filename, page, and chunk)
- Generate document summaries
- Extract important keywords
- View document statistics
- Handle empty documents and unsupported files

## 🧠 How RAG Works

Retrieval-Augmented Generation (RAG) combines document retrieval with language model generation.

### Workflow

1. Upload a PDF or TXT document.
2. Extract the text from the document.
3. Split the text into smaller chunks.
4. Convert the chunks into embeddings.
5. Store embeddings in ChromaDB.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant document chunks.
8. Pass the retrieved context to the Groq LLM.
9. Generate an answer based on the retrieved content.
10. Display the answer with source references.

This approach helps the application answer questions using the uploaded documents instead of relying only on the LLM's general knowledge.

## 🏗️ System Architecture

```text
User
  |
  v
Streamlit Interface
  |
  v
Upload PDF / TXT Document
  |
  v
Text Extraction
  |
  v
Text Cleaning and Chunking
  |
  v
Hugging Face Embeddings
  |
  v
ChromaDB Vector Store
  |
  v
User Question
  |
  v
Similarity Search (Top-K)
  |
  v
Relevant Document Chunks
  |
  v
Groq LLM
  |
  v
Answer + Source References
  |
  v
Streamlit Interface
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web interface |
| PyMuPDF | Extract text from PDF documents |
| LangChain | Text splitting and RAG components |
| Hugging Face Embeddings | Convert text into numerical vectors |
| ChromaDB | Store and search document embeddings |
| Groq | LLM-based answer generation |
| python-dotenv | Manage environment variables |

## 📂 Project Structure

```text
DocumentIntelligence/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── data/
│   └── uploads/
│
├── chroma_db/
│
└── src/
    ├── document_loader.py
    ├── text_processing.py
    ├── embeddings.py
    ├── vector_store.py
    ├── rag_pipeline.py
    ├── summarizer.py
    └── utils.py
```

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/HariniRavi26/DocumentIntelligence.git
cd DocumentIntelligence
```

### 2. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Do not upload your API key to GitHub.

### 5. Run the application

```powershell
streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 🔍 Key Concepts

### RAG

Retrieves relevant information from documents and uses it as context for generating answers.

### Embeddings

Converts text into numerical vectors that represent its meaning.

### ChromaDB

Stores embeddings and supports similarity-based document retrieval.

### Groq LLM

Generates responses using the relevant document context.

### Source References

Displays document metadata such as filename, page, and chunk to help identify the source of the answer.

## ☁️ Streamlit Cloud Deployment

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select the `main` branch.
5. Set the main file to `app.py`.
6. Add the `GROQ_API_KEY` in Streamlit Secrets.
7. Deploy the application.
8. Copy the deployed application URL.
9. Update the Live Demo link in this README.


## 🎯 Project Objective

The objective of this project is to develop a document-based question-answering system using RAG, embeddings, vector databases, and large language models.

## 👩‍💻 Author

**Harini Ravi**

- GitHub: [HariniRavi26](https://github.com/HariniRavi26)
- LinkedIn: [Harini Ravi](https://linkedin.com/in/harini-ravi-658a68315)
