CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_DB_PATH = ".rag_index"

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path("ai_cli/data/rag")

INDEX_DIR = BASE_DIR / "indexes"
DOCS_DIR = BASE_DIR / "documents"
METADATA_DIR = BASE_DIR / "metadata"

INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = METADATA_DIR / "metadata.json"