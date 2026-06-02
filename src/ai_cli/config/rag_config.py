"""
Advanced RAG configuration and utilities.

This file centralizes configuration and small helper utilities used across the
RAG pipeline: chunking, embedding options, and vector DB (FAISS) configuration.

How to use:
- Import the default_config object for global defaults.
- Call default_config.ensure_dirs() early in your app to create folders.
- Use chunk_text(...) to split long documents into overlapping chunks suitable
    for embedding.
- The VectorStoreConfig describes which vector DB to use; currently it includes
    file paths for FAISS. The actual indexing/querying code should use these
    values (see comments below).

Notes:
- The chunking implementation here is tokenizer-agnostic (splits on words or
    sentence boundaries). For production use, replace `simple_tokenize` with a
    tokenizer that matches your embedding model (e.g., tiktoken for OpenAI,
    or a sentence splitter for better boundaries).
- Embedding integration and FAISS indexing/querying are intentionally kept as
    small helpers/stubs. They document how to wire things up; the heavy lifting
    should be implemented where the project's dependencies are managed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Dict
import json
import logging

logger = logging.getLogger(__name__)


# Default folder layout (relative to project root or running CWD).
BASE_DIR = Path("ai_cli/data/rag")
INDEX_DIR = BASE_DIR / "indexes"
DOCS_DIR = BASE_DIR / "documents"
METADATA_DIR = BASE_DIR / "metadata"

# Ensure directories exist on import (safe no-op if already present).
INDEX_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)


def simple_tokenize(text: str) -> List[str]:
        """
        Fast, dependency-free tokenization: split on whitespace.
        Replace with a model/tokenizer-aware implementation for production.
        """
        return text.split()


def sentence_split(text: str) -> List[str]:
        """
        Very simple sentence splitter using punctuation. Use a proper sentence
        tokenizer (e.g., nltk.sent_tokenize or spacy) for robust behavior.
        """
        import re

        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p for p in parts if p]


def chunk_text(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        strategy: str = "sliding",
        tokenizer: Optional[Callable[[str], List[str]]] = None,
) -> List[str]:
        """
        Chunk `text` into pieces suitable for embedding.

        Parameters:
        - text: the full document text
        - chunk_size: target number of tokens/words per chunk
        - chunk_overlap: overlap between chunks (in tokens/words)
        - strategy: "sliding" (default) or "sentence" (attempt to split by sentences)
        - tokenizer: function(text) -> list(tokens). If None, uses simple_tokenize.

        Returns:
        - list of text chunks
        """
        tokenizer = tokenizer or simple_tokenize

        if strategy == "sentence":
                sentences = sentence_split(text)
                chunks = []
                current = []
                current_len = 0
                for s in sentences:
                        toks = tokenizer(s)
                        if current_len + len(toks) <= chunk_size:
                                current.append(s)
                                current_len += len(toks)
                        else:
                                if current:
                                        chunks.append(" ".join(current))
                                # if single sentence longer than chunk_size, fallback to word-splitting
                                if len(toks) > chunk_size:
                                        words = tokenizer(s)
                                        i = 0
                                        while i < len(words):
                                                part = words[i : i + chunk_size]
                                                chunks.append(" ".join(part))
                                                i += chunk_size - chunk_overlap
                                        current = []
                                        current_len = 0
                                else:
                                        current = [s]
                                        current_len = len(toks)
                if current:
                        chunks.append(" ".join(current))
                return chunks

        # sliding window by tokens (default)
        tokens = tokenizer(text)
        if not tokens:
                return []

        chunks = []
        i = 0
        while i < len(tokens):
                chunk_tokens = tokens[i : i + chunk_size]
                chunks.append(" ".join(chunk_tokens))
                if i + chunk_size >= len(tokens):
                        break
                i += chunk_size - chunk_overlap
        return chunks


@dataclass
class VectorStoreConfig:
        """
        Minimal vector-store configuration.

        - store_type: currently "faiss" or "memory" (in-memory fallback).
        - faiss_index_path: path to persist FAISS index (if using FAISS).
        - recreate: whether to recreate index on save/load (useful in tests).
        - metric: similarity metric used by index (e.g., "cosine", "l2").
        """
        store_type: str = "faiss"
        faiss_index_path: Path = INDEX_DIR / "index.faiss"
        recreate: bool = False
        metric: str = "cosine"


@dataclass
class EmbeddingConfig:
        """
        Embedding model parameters.
        - model_name: identifier for embedding model/library (sentence-transformers,
            OpenAI embedding name, etc.)
        - batch_size: recommended batch size when batching embedding requests.
        - normalize: whether to l2-normalize embeddings for cosine similarity.
        """
        model_name: str = "all-MiniLM-L6-v2"
        batch_size: int = 32
        normalize: bool = True


@dataclass
class RAGConfig:
        """
        Central RAG configuration.
        """
        base_dir: Path = BASE_DIR
        index_dir: Path = INDEX_DIR
        docs_dir: Path = DOCS_DIR
        metadata_dir: Path = METADATA_DIR

        chunk_size: int = 500
        chunk_overlap: int = 50
        chunk_strategy: str = "sliding"  # or "sentence"

        top_k: int = 5

        embedding: EmbeddingConfig = EmbeddingConfig()
        vector_store: VectorStoreConfig = VectorStoreConfig()

        metadata_path: Path = METADATA_DIR / "metadata.json"

        def ensure_dirs(self) -> None:
                """Create configured directories if they don't exist."""
                for p in (self.base_dir, self.index_dir, self.docs_dir, self.metadata_dir):
                        p.mkdir(parents=True, exist_ok=True)

        def to_dict(self) -> Dict:
                """Return serializable dict representation."""
                d = asdict(self)
                # convert paths to strings
                for k in ("base_dir", "index_dir", "docs_dir", "metadata_dir", "metadata_path"):
                        if k in d and isinstance(d[k], Path):
                                d[k] = str(d[k])
                # nested dataclasses already converted by asdict
                return d

        def save(self, path: Optional[Path] = None) -> None:
                """Save config as JSON for reproducibility."""
                path = Path(path) if path else self.metadata_dir / "rag_config.json"
                self.ensure_dirs()
                with open(path, "w", encoding="utf-8") as f:
                        json.dump(self.to_dict(), f, indent=2)
                logger.info("Saved RAG config to %s", path)

        @classmethod
        def load(cls, path: Path) -> "RAGConfig":
                """Load a config previously saved with save()."""
                with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                # convert path strings back to Path objects where relevant
                for key in ("base_dir", "index_dir", "docs_dir", "metadata_dir", "metadata_path"):
                        if key in data and data[key]:
                                data[key] = Path(data[key])
                # reconstruct nested dataclasses
                emb = EmbeddingConfig(**data.get("embedding", {}))
                vs = VectorStoreConfig(**data.get("vector_store", {}))
                # take remaining keys for RAGConfig
                cfg_kwargs = {
                        k: v
                        for k, v in data.items()
                        if k
                        in {
                                "base_dir",
                                "index_dir",
                                "docs_dir",
                                "metadata_dir",
                                "chunk_size",
                                "chunk_overlap",
                                "chunk_strategy",
                                "top_k",
                                "metadata_path",
                        }
                }
                cfg = cls(**cfg_kwargs, embedding=emb, vector_store=vs)
                return cfg

        # Small helper documenting where to plug in vector DB code.
        def faiss_index_path(self) -> Path:
                """Convenience: path to FAISS index file."""
                return Path(self.vector_store.faiss_index_path)


# module-level default config instance
default_config = RAGConfig()


# Lightweight example helper showing how embedding + FAISS wiring might look.
# Real implementations should live in modules that declare dependencies (faiss,
# sentence-transformers, OpenAI SDK, etc.).
def build_faiss_index_stub(embeddings: Iterable[List[float]], ids: Optional[Iterable[str]] = None):
        """
        Stub demonstrating the expected inputs to build a FAISS index.
        - embeddings: iterable of embedding vectors (list/array-like)
        - ids: optional iterable of ids aligned to embeddings

        The body is intentionally left as a stub to avoid hard dependency on faiss.
        Replace with real code such as:
            import faiss
            xb = np.asarray(list(embeddings)).astype('float32')
            index = faiss.IndexFlatIP(d)  # or other index type
            index.add(xb)
            faiss.write_index(index, str(default_config.faiss_index_path()))
        """
        raise NotImplementedError("Implement FAISS index building in your indexer module.")


def query_vector_store_stub(query_embedding: List[float], top_k: Optional[int] = None):
        """
        Stub demonstrating vector DB query signature.
        - query_embedding: single embedding vector (list/array-like)
        - top_k: number of nearest neighbors to return (defaults to config.top_k)

        Expected return: list of (id, score) pairs or documents depending on
        your retrieval layer.
        """
        raise NotImplementedError("Implement vector store querying in your retriever module.")


# Expose commonly-used constants for backwards compatibility with older code
CHUNK_SIZE = default_config.chunk_size
CHUNK_OVERLAP = default_config.chunk_overlap
TOP_K = default_config.top_k
EMBEDDING_MODEL = default_config.embedding.model_name
FAISS_INDEX_PATH = default_config.faiss_index_path()
METADATA_PATH = default_config.metadata_path