from __future__ import annotations

import math
import os
from typing import Any

import cohere

# IMPORTANT:
# Use relative import to avoid CI/package import resolution issues
from .base import BaseProvider


class CohereProvider(BaseProvider):
    """
    Cohere provider with optional RAG support.

    Design contract aligned with OpenAI/Gemini providers:
    - append-only in-memory vector store
    - chunk → embed → store lifecycle
    - cosine similarity retrieval
    """

    def __init__(self, *, rag_enabled: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.rag_enabled = rag_enabled

        api_key = (
                   kwargs.get("api_key")
                   or getattr(self, "api_key", None)
                   or os.getenv("COHERE_API_KEY")
                  )

        if not api_key:
            raise ValueError("COHERE_API_KEY is required")

        self.client = cohere.Client(api_key)

        # ----------------------------
        # In-memory vector store
        # ----------------------------
        self._documents: list[str] = []
        self._vectors: list[list[float]] = []
        self._metadata: list[dict[str, Any]] = []

    # =========================================================
    # Chat
    # =========================================================

    def send(self, prompt: str, **kwargs) -> str:
        """
        Chat with optional RAG context injection.
        """

        if not self.rag_enabled:
            return self._chat(prompt)

        context = self.retrieve(prompt, top_k=5)

        if context:
            context_text = "\n\n".join(
                f"[{i}] {item['text']}"
                for i, item in enumerate(context)
            )

            prompt = (
                "Use the following context to answer the question.\n\n"
                f"{context_text}\n\n"
                f"Question: {prompt}"
            )

        return self._chat(prompt)

    def _chat(self, prompt: str) -> str:
        """
        Low-level Cohere chat call.
        """

        resp = self.client.chat(
            message=prompt,
        )
        return resp.text

    # =========================================================
    # Embeddings
    # =========================================================

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """
        Cohere embedding wrapper.
        """
        if not texts:
            return []

        resp = self.client.embed(
            texts=texts,
            input_type="search_document",
        )
        return resp.embeddings

    # =========================================================
    # RAG ingestion
    # =========================================================

    def upsert_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Chunk → embed → store pipeline.

        Contract:
        - chunk-level storage
        - strict alignment between vectors/documents/metadata
        """

        if not texts:
            return

        chunk_texts: list[str] = []
        chunk_meta: list[dict[str, Any]] = []

        # ----------------------------
        # Chunk documents
        # ----------------------------
        for doc_idx, doc_text in enumerate(texts):
            doc_metadata = (
                metadatas[doc_idx]
                if metadatas and doc_idx < len(metadatas)
                else {}
            )

            for chunk_idx, chunk in enumerate(self._chunk_text(doc_text)):
                chunk_texts.append(chunk)

                chunk_meta.append({
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                    "metadata": doc_metadata,
                })

        if not chunk_texts:
            return

        # ----------------------------
        # Embed chunks
        # ----------------------------
        embeddings = self._embed(chunk_texts)

        if len(embeddings) != len(chunk_texts):
            raise RuntimeError(
                f"Embedding mismatch: {len(embeddings)} != {len(chunk_texts)}"
            )

        # ----------------------------
        # Store (append-only contract)
        # ----------------------------
        for vec, txt, meta in zip(embeddings, chunk_texts, chunk_meta, strict=False):
            self._vectors.append(vec)
            self._documents.append(txt)
            self._metadata.append(meta)

    # =========================================================
    # Retrieval
    # =========================================================

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Cosine similarity search over in-memory embeddings.
        """

        if not self._vectors:
            return []

        q_vec = self._embed([query])[0]

        scored: list[dict[str, Any]] = []

        for i, vec in enumerate(self._vectors):
            score = self._cosine_similarity(q_vec, vec)

            scored.append({
                "text": self._documents[i],
                "metadata": self._metadata[i],
                "score": score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """
        Stable cosine similarity implementation.
        """

        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        return dot / (norm_a * norm_b + 1e-8)

    # =========================================================
    # Utility
    # =========================================================

    def clear_index(self) -> None:
        """
        Reset in-memory vector store.
        """
        self._documents.clear()
        self._vectors.clear()
        self._metadata.clear()

    def query_documents(self, query: str, top_k: int = 5):
        if not self.vector_store:
            raise ValueError("RAG is not enabled for this provider")

        return self.vector_store.search(query, top_k=top_k)