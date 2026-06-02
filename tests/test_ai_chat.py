# ai_cli/rag.py
from __future__ import annotations
import math
import hashlib
from typing import List, Tuple, Iterable, Dict

# Chunking
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Simple deterministic chunker that splits on whitespace.
        - chunk_size: approximate number of characters per chunk
        - overlap: number of characters to overlap between chunks
        """
        if not isinstance(text, str) or not text.strip():
                return []
        text = text.strip()
        chunks: List[str] = []
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
                # avoid infinite loop: if end == n break
                if end >= n:
                        break
        return chunks

# Deterministic "embedding" using hash -> fixed-dim float vector in [-1,1]
def _text_to_embedding(text: str, dim: int = 64) -> List[float]:
        if not isinstance(text, str):
                text = ""
        h = hashlib.sha256(text.encode("utf8")).digest()
        # expand or repeat hash bytes to fill dim
        vec: List[float] = []
        i = 0
        while len(vec) < dim:
                b = h[i % len(h)]
                # convert byte [0,255] to float [-1,1]
                vec.append((b / 255.0) * 2.0 - 1.0)
                i += 1
        return vec[:dim]

def embed_texts(texts: Iterable[str], dim: int = 64) -> List[List[float]]:
        return [_text_to_embedding(t or "", dim=dim) for t in texts]

# Cosine similarity
def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
                return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for x, y in zip(a, b):
                dot += x * y
                na += x * x
                nb += y * y
        if na == 0 or nb == 0:
                return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

# Simple in-memory vector DB
class VectorStore:
        def __init__(self, dim: int = 64):
                self.dim = dim
                self._docs: Dict[str, str] = {}
                self._embeddings: Dict[str, List[float]] = {}

        def add(self, doc_id: str, text: str) -> None:
                emb = _text_to_embedding(text, dim=self.dim)
                self._docs[doc_id] = text
                self._embeddings[doc_id] = emb

        def add_many(self, items: Iterable[Tuple[str, str]]) -> None:
                for doc_id, text in items:
                        self.add(doc_id, text)

        def query(self, query_text: str, top_k: int = 3, min_score: float = 0.0) -> List[Tuple[str, str, float]]:
                q_emb = _text_to_embedding(query_text, dim=self.dim)
                results: List[Tuple[str, str, float]] = []
                for doc_id, emb in self._embeddings.items():
                        score = _cosine(q_emb, emb)
                        if score >= min_score:
                                results.append((doc_id, self._docs[doc_id], score))
                results.sort(key=lambda t: t[2], reverse=True)
                return results[:top_k]

        def all_docs(self) -> Dict[str, str]:
                return dict(self._docs)

# Convenience: build a VectorStore from a long document (chunked)
def build_store_from_text(doc_id_prefix: str, text: str, chunk_size: int = 500, overlap: int = 50, dim: int = 64) -> VectorStore:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        store = VectorStore(dim=dim)
        for i, c in enumerate(chunks):
                store.add(f"{doc_id_prefix}-{i}", c)
        return store
