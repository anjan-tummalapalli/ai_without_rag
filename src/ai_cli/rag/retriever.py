from __future__ import annotations

from pathlib import Path

# Retrieval config
TOP_K = 5
CHUNK_SIZE = 1000          # characters per chunk (tune for your model / context window)
CHUNK_OVERLAP = 200        # characters overlap between chunks
EMBEDDING_DIMS = 768       # default dim used by the real model; used by fallback
VECTOR_STORE_PATH = Path.home() / ".ai_cli" / "rag_vectors.pkl"

# Vector store backend: "faiss" or "memory"
VECTORDB_BACKEND = "faiss"
