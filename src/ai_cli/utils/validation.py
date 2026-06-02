from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Dict, Any
import logging

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
    reasons: List[str] = field(default_factory=list)


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
        reasons: List[str] = []

        if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
            score += 0.4
            reasons.append("response too short")

        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.2
                reasons.append(f"suspicious phrase: {pattern}")

        if "TODO" in response:
            score += 0.3
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
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

    chunks: List[str] = []
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

    def embed_texts(self, texts: Sequence[str]) -> List:
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


class VectorStore:
    """
    Simple vector store abstraction with optional FAISS backend.

    Usage:
        store = VectorStore(dim=embedder.dim)
        store.add(texts, embeddings, metadatas)
        results = store.search(query_embedding, top_k=5)
    """

    def __init__(self, dim: int = _DEFAULT_EMBED_DIM, use_faiss: bool = True):
        self.dim = dim
        self._use_faiss = use_faiss and _HAS_FAISS
        self._ids: List[int] = []
        self._texts: List[str] = []
        self._metadatas: List[Optional[Dict[str, Any]]] = []
        self._next_id = 0

        if self._use_faiss:
            self._index = faiss.IndexFlatIP(dim)  # inner product; embeddings should be normalized
            self._id_to_pos: Dict[int, int] = {}
        else:
            self._index = None
            self._embeddings = []  # store numpy arrays or lists

    def add(self, texts: Sequence[str], embeddings: Sequence, metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None):
        """
        Add documents and corresponding embeddings to the store.

        Args:
            texts: sequence of document text strings
            embeddings: sequence of vectors (numpy arrays or lists)
            metadatas: optional sequence of metadata dicts
        """
        if metadatas is None:
            metadatas = [None] * len(texts)

        for text, emb, md in zip(texts, embeddings, metadatas):
            id_ = self._next_id
            self._next_id += 1
            self._ids.append(id_)
            self._texts.append(text)
            self._metadatas.append(md)
            if self._use_faiss:
                vec = np.asarray(emb, dtype="float32")
                # normalize for inner product similarity
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._index.add(np.expand_dims(vec, axis=0))
                self._id_to_pos[id_] = len(self._ids) - 1
            else:
                self._embeddings.append(np.asarray(emb, dtype="float32") if np is not None else emb)

    def _search_in_memory(self, query_vec, top_k: int):
        if np is None:
            raise RuntimeError("numpy required for in-memory search")
        q = np.asarray(query_vec, dtype="float32")
        qnorm = np.linalg.norm(q)
        if qnorm > 0:
            q = q / qnorm
        sims = []
        for idx, emb in enumerate(self._embeddings):
            emb_arr = np.asarray(emb, dtype="float32")
            embnorm = np.linalg.norm(emb_arr)
            if embnorm > 0:
                emb_arr = emb_arr / embnorm
            sims.append(float(np.dot(q, emb_arr)))
        # argsort descending
        ranked_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [(self._ids[i], sims[i], self._texts[i], self._metadatas[i]) for i in ranked_idx]

    def search(self, query_embedding, top_k: int = 5) -> List[Tuple[int, float, str, Optional[Dict[str, Any]]]]:
        """
        Search the store for nearest neighbors.

        Returns:
            list of tuples: (doc_id, score, text, metadata)
        """
        if self._use_faiss:
            if np is None:
                raise RuntimeError("numpy required for faiss backend")
            q = np.asarray(query_embedding, dtype="float32")
            qnorm = np.linalg.norm(q)
            if qnorm > 0:
                q = q / qnorm
            D, I = self._index.search(np.expand_dims(q, axis=0), top_k)
            results = []
            for score, pos in zip(D[0], I[0]):
                if pos < 0:
                    continue
                doc_id = self._ids[pos]
                results.append((doc_id, float(score), self._texts[pos], self._metadatas[pos]))
            return results
        else:
            return self._search_in_memory(query_embedding, top_k=top_k)


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

    def __init__(self, embedder: Optional[Embedder] = None, vector_store: Optional[VectorStore] = None):
        self.embedder = embedder or Embedder()
        self.store = vector_store or VectorStore(dim=self.embedder.dim)

    def index_document(self, doc_text: str, chunk_size: int = 500, overlap: int = 50, metadata: Optional[Dict[str, Any]] = None):
        chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=overlap)
        embeddings = self.embedder.embed_texts(chunks)
        metadatas = [{"chunk_index": i, **(metadata or {})} for i in range(len(chunks))]
        self.store.add(chunks, embeddings, metadatas)

    def query(self, query_text: str, top_k: int = 5):
        query_emb = self.embedder.embed_texts([query_text])[0]
        hits = self.store.search(query_emb, top_k=top_k)
        # Return texts and scores for convenience
        return [{"id": h[0], "score": h[1], "text": h[2], "metadata": h[3]} for h in hits]
