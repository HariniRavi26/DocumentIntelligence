import os
import logging
from dotenv import load_dotenv

# Load variables from a local .env file (GROQ_API_KEY, etc.)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where uploaded files are temporarily saved on disk
UPLOAD_DIR = os.path.join("data", "uploads")

# Where ChromaDB persists its vector index
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "document_intelligence"

# Embedding model: small, fast, runs on CPU, ~80MB download
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
# Chunking parameters (characters, not tokens)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Retrieval
DEFAULT_TOP_K = 4

# Supported upload types
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Max upload size we bother processing (in MB) - purely a sanity guard
MAX_FILE_SIZE_MB = 50


def get_groq_api_key() -> str | None:
    """Fetch the Groq API key from environment / .env file."""
    return os.getenv("GROQ_API_KEY")


def ensure_dirs() -> None:
    """Make sure the folders we write to actually exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a simply-configured logger so errors show up in the terminal."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
