
# EMBEDDING_MODEL = "text-embedding-ada-002"
# LLM_MODEL = "gpt-4o"
# VECTOR_DB_PATH = "vectorstore/faiss_index.pkl"

import os

# Optional: Load local .env file when running locally (for development only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load configuration from environment variables or fallback to defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # REQUIRED in Streamlit Secrets
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vectorstore/faiss_index.pkl")

# Safety check for missing key (optional for debugging)
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found. Please set it in environment or .env file.")

