from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Optional deps - fallbacks handled below to avoid hard dependency
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected but handle gracefully
    np = None  # type: ignore

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_SBT = True
except Exception:
    _HAS_SBT = False

# Local exceptions
from ai_cli.core.exceptions import ResponseValidationError

MIN_RESPONSE_LENGTH = 5
_DEFAULT_EMBED_DIM = 768

log = logging.getLogger(__name__)


@dataclass
class HallucinationResult:
    """Result of hallucination risk evaluation.

    score: normalized risk in [0.0, 1.0]
    passed: True when risk is below threshold (default threshold: 0.5)
    reasons: human-readable labels for triggers
    """

    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


class HallucinationDetector:
    """
    Heuristic-based hallucination risk estimator.

    Purpose:
        Lightweight triage to flag responses that warrant further verification.

    Signals:
        - very short responses
        - matches to suspicious phrases
        - presence of placeholder tokens like "TODO"
    """

    SUSPICIOUS_PATTERNS = [
        r"100% accurate",
        r"guaranteed",
        r"always works",
        r"never fails",
        r"trust me",
    ]

    def evaluate(self, response: str) -> HallucinationResult:
        """Evaluate response for hallucination risk."""
        score = 0.0
        reasons: list[str] = []

        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            score += 0.4
            reasons.append("response too short")

        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.lower() in response.lower():
                score += 0.2
                reasons.append(f"suspicious phrase: {pattern}")

        if "TODO" in response:
            score += 0.6
            reasons.append("placeholder content detected")

        score = min(score, 1.0)
        return HallucinationResult(score=score, passed=score < 0.5, reasons=reasons)


class ResponseValidator:
    """Simple response validation helper."""

    def validate(self, response: str) -> None:
        """Validate the response string.

        Raises:
            ResponseValidationError: If response is empty or too short.
        """
        if not response:
            raise ResponseValidationError("empty response")
        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            raise ResponseValidationError("response too short")


# ---------------------------
# Advanced RAG utilities
# ---------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Chunk a long text into overlapping windows.

    Args:
        text: input string to chunk.
        chunk_size: max characters per chunk.
        overlap: number of characters to overlap between neighboring chunks.

    Returns:
        list of text chunks (strings).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


class Embedder:
    """
    Minimal pluggable embedder.

    Backends supported (if available):
      - 'sentence_transformer': uses sentence-transformers SentenceTransformer
      - 'in-memory': a simple randomized/deterministic embedder for testing

    For production use, prefer a proper embeddings provider (OpenAI, HuggingFace, etc.)
    and implement a backend that calls that service.
    """

    def __init__(self, backend: str = "sentence_transformer", model_name: str = "all-MiniLM-L6-v2"):
        self.backend = backend
        self.model_name = model_name
        self._model = None
        self._dim = _DEFAULT_EMBED_DIM
        if backend == "sentence_transformer" and _HAS_SBT:
            try:
                self._model = SentenceTransformer(model_name)
                self._dim = self._model.get_sentence_embedding_dimension()
            except Exception as exc:
                log.warning("Failed to load SentenceTransformer model: %s", exc)
                self._model = None

        if self._model is None:
            # fallback deterministic pseudo-embedder (not semantically meaningful)
            self.backend = "in-memory"
            self._dim = 256

    @property
    def dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> list:
        """
        Embed a list of texts.

        Returns:
            List of vectors (numpy arrays if numpy present, else Python lists).
        """
        texts = list(texts)
        if self.backend == "sentence_transformer" and self._model is not None:
            vectors = self._model.encode(texts, show_progress_bar=False)
            return [np.asarray(v) for v in vectors] if np is not None else [list(v) for v in vectors]

        # deterministic fallback: hash-based embedding
        out = []
        for i, t in enumerate(texts):
            v = [(hash(t + str(i + j)) % 1000) / 1000.0 for j in range(self._dim)]
            if np is not None:
                out.append(np.array(v, dtype=float))
            else:
                out.append(v)
        return out

# ---------------------------
# Example glue: RAG helper
# ---------------------------

class RAGHelper:
    """
    Convenience helper that wires chunking, embedding, and vector store together.

    Typical workflow:
        - chunk documents with chunk_text
        - embed chunks with Embedder.embed_texts
        - add to VectorStore
        - for a query: embed the query and call VectorStore.search

    Notes:
        This component is intentionally small and backend-agnostic. For production,
        add batching, persistence, and robust error handling.
    """

    def __init__(self, embedder: Embedder | None = None, vector_store: VectorStore | None = None):
        self.embedder = embedder or Embedder()
        self.store = None

    def index_document(self, doc_text: str, chunk_size: int = 500, overlap: int = 50, metadata: dict[str, Any] | None = None):
        chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=overlap)
        embeddings = self.embedder.embed_texts(chunks)
        metadatas = [{"chunk_index": i, **(metadata or {})} for i in range(len(chunks))]
        self.store.add(chunks, embeddings, metadatas)

    def query(self, query_text: str, top_k: int = 5):
        query_emb = self.embedder.embed_texts([query_text])[0]
        hits = self.store.search(query_emb, top_k=top_k)
        # Return texts and scores for convenience
        return [{"id": h[0], "score": h[1], "text": h[2], "metadata": h[3]} for h in hits]