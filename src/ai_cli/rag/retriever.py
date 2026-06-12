"""Semantic retrieval over a FAISS-backed vector store."""

from __future__ import annotations

from typing import Callable, List, Optional

from ai_cli.config.rag_config import TOP_K
from ai_cli.rag.embeddings import EmbeddingGenerator
from ai_cli.rag.models import Chunk, RetrievalResult
from ai_cli.rag.vector_store import VectorStore


class Retriever:
    """Embed queries and search a ``VectorStore`` for relevant chunks."""

    def __init__(
        self,
        store: VectorStore,
        embedder: EmbeddingGenerator,
        top_k: int = TOP_K,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
    ) -> List[RetrievalResult]:
        k = top_k if top_k is not None else self.top_k
        query_vec = self.embedder.embed_text(query)
        hits = self.store.search(query_vec, top_k=k, filter_fn=filter_fn)
        return [
            RetrievalResult(chunk=item["chunk"], score=item.get("score", 0.0))
            for item in hits
        ]

    def build_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        results = self.retrieve(query, top_k=top_k)
        return separator.join(r.chunk.text for r in results)
