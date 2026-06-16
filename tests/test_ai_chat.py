# ai_cli/rag.py
from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Iterable


# Chunking
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """
        Simple deterministic chunker that splits on whitespace.
        - chunk_size: approximate number of characters per chunk
        - overlap: number of characters to overlap between chunks
        """
        if not isinstance(text, str) or not text.strip():
                return []
        text = text.strip()
        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
                end = start + chunk_size
                # try to avoid cutting words: move end back to last space if possible
                if end < n:
                        last_space = text.rfind(" ", start, end)
                        if last_space > start:
                                end = last_space
                chunk = text[start:end].strip()
                if chunk:
                        chunks.append(chunk)
                start = end - overlap
                if start < 0:
                        start = 0
                if end >= n:
                        break
        return chunks

# Deterministic "embedding" using hash -> fixed-dim float vector in [-1,1]
def _text_to_embedding(text: str, dim: int = 64) -> list[float]:
        if not isinstance(text, str):
                text = ""
        h = hashlib.sha256(text.encode("utf8")).digest()
        vec: list[float] = []
        i = 0
        # iterate over hash bytes, repeat as needed
        while len(vec) < dim:
                b = h[i % len(h)]
                vec.append((b / 255.0) * 2.0 - 1.0)
                i += 1
        return vec[:dim]

def embed_texts(texts: Iterable[str], dim: int = 64) -> list[list[float]]:
        return [_text_to_embedding(t or "", dim=dim) for t in texts]

# Cosine similarity helpers
def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=False))

def _norm(a: list[float]) -> float:
        s = sum(x * x for x in a)
        return math.sqrt(s) if s > 0.0 else 0.0

# Simple in-memory vector DB with small optimizations
class VectorStore:
        def __init__(self, dim: int = 64):
                self.dim = dim
                self._docs: dict[str, str] = {}
                self._embeddings: dict[str, list[float]] = {}
                self._norms: dict[str, float] = {}

        def add(self, doc_id: str, text: str) -> None:
                emb = _text_to_embedding(text, dim=self.dim)
                self._docs[doc_id] = text
                self._embeddings[doc_id] = emb
                self._norms[doc_id] = _norm(emb)

        def add_many(self, items: Iterable[tuple[str, str]]) -> None:
                for doc_id, text in items:
                        self.add(doc_id, text)

        def query(self, query_text: str, top_k: int = 3, min_score: float = 0.0) -> list[tuple[str, str, float]]:
                """
                Returns top_k (doc_id, text, score) ordered by score desc.
                Uses a min-heap of size top_k for O(n log k) selection.
                """
                q_emb = _text_to_embedding(query_text, dim=self.dim)
                q_norm = _norm(q_emb)
                if q_norm == 0.0:
                        return []

                heap: list[tuple[float, str, str]] = []  # (score, doc_id, text)
                for doc_id, emb in self._embeddings.items():
                        doc_norm = self._norms.get(doc_id, 0.0)
                        if doc_norm == 0.0:
                                continue
                        score = _dot(q_emb, emb) / (q_norm * doc_norm)
                        if score < min_score:
                                continue
                        if len(heap) < top_k:
                                heapq.heappush(heap, (score, doc_id, self._docs[doc_id]))
                        else:
                                # heap[0] is smallest score in heap, replace if current score is higher
                                if score > heap[0][0]:
                                        heapq.heapreplace(heap, (score, doc_id, self._docs[doc_id]))

                # convert heap to sorted list (desc)
                results = [(doc_id, text, score) for score, doc_id, text in heap]
                results.sort(key=lambda t: t[2], reverse=True)
                return results

        def all_docs(self) -> dict[str, str]:
                return dict(self._docs)

# Convenience: build a VectorStore from a long document (chunked)
def build_store_from_text(doc_id_prefix: str, text: str, chunk_size: int = 500, overlap: int = 50, dim: int = 64) -> VectorStore:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        store = VectorStore(dim=dim)
        for i, c in enumerate(chunks):
                store.add(f"{doc_id_prefix}-{i}", c)
        return store
