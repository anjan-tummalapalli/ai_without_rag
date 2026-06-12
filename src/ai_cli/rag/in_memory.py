"""Lightweight in-memory RAG pipeline for CLI and quick prototyping."""

from __future__ import annotations

import hashlib
import heapq
import math
import struct
import uuid
from typing import Any, Dict, List, Optional, Sequence


class InMemoryRAGPipeline:
    """
    In-memory RAG pipeline with deterministic hash-based embeddings.

    Suitable for CLI demos and tests. Swap in ``EmbeddingGenerator`` +
    ``VectorStore`` for production semantic search.
    """

    def __init__(self, embed_dim: int = 128) -> None:
        self.embed_dim = embed_dim
        self._store: List[Dict[str, Any]] = []

    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        overlap = max(0, overlap)
        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= length:
                break
            start = max(0, end - overlap)
        return chunks

    def _embed_one(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding from SHA-256 (not semantic)."""
        base = hashlib.sha256(text.encode("utf-8")).digest()
        vec: List[float] = []
        for i in range(self.embed_dim):
            digest = hashlib.sha256(base + i.to_bytes(2, "little")).digest()
            val = struct.unpack(">I", digest[:4])[0]
            vec.append((val / 0xFFFFFFFF) * 2.0 - 1.0)
        return vec

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        norma = math.fsum(x * x for x in a)
        normb = math.fsum(y * y for y in b)
        if norma <= 0 or normb <= 0:
            return 0.0
        return num / (math.sqrt(norma) * math.sqrt(normb))

    def upsert_documents(
        self,
        doc_texts: Sequence[str],
        doc_ids: Optional[Sequence[str]] = None,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        if doc_ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in doc_texts]
        for doc_id, text in zip(doc_ids, doc_texts):
            chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk, emb in zip(chunks, self.embed_texts(chunks)):
                self._store.append(
                    {
                        "id": str(uuid.uuid4()),
                        "doc_id": doc_id,
                        "chunk": chunk,
                        "embedding": emb,
                        "meta": {"length": len(chunk)},
                    }
                )

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        if not self._store or top_k <= 0:
            return ""
        q_emb = self._embed_one(query)
        scored = (
            (self._cosine(q_emb, entry["embedding"]), entry)
            for entry in self._store
        )
        top = [
            entry
            for score, entry in heapq.nlargest(top_k, scored, key=lambda x: x[0])
            if score > 0.0
        ]
        return "\n\n---\n\n".join(e["chunk"] for e in top)

    def __len__(self) -> int:
        return len(self._store)


# Backward-compatible alias used by the CLI
RAGPipeline = InMemoryRAGPipeline
