"""
Cohere provider for ai_cli with Advanced RAG support.

Integrates Cohere LLMs via the Cohere SDK and adds basic RAG:
- chunking (configurable size/overlap)
- embeddings via Cohere Embeddings API
- simple in-memory vector DB
- cosine similarity retrieval
- automatic context augmentation for chat requests

Environment:
    COHERE_API_KEY must be set.
"""
from __future__ import annotations

import importlib
import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import cohere  # type: ignore

from ai_cli.core.exceptions import (  # pylint: disable=C0413
    ProviderRequestError,
)
from ai_cli.providers.base import AIProvider  # pylint: disable=C0413


# pylint: disable=too-many-instance-attributes
class CohereProvider(AIProvider):
    """Cohere AI provider with optional RAG.

    Parameters
    ----------
    model:
        Cohere model name, default ``"command-r"``.
    api_key:
        Cohere API key; falls back to the ``COHERE_API_KEY`` env var.
    rag_enabled:
        Enable Retrieval-Augmented Generation.
    embedding_model:
        Embedding model to use, default ``"embed-english-v2.0"``.
    chunk_size:
        Character chunk size for document splitting, default ``500``.
    chunk_overlap:
        Character overlap between consecutive chunks, default ``50``.
    vector_store_backend:
        Backend for the vector store, default ``"memory"``.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        rag_enabled: bool = False,
        embedding_model: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_backend: str = "memory",
        **kwargs,
    ) -> None:
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ProviderRequestError(
                "COHERE_API_KEY environment variable is not set"
            )

        # Import Cohere SDK at runtime to keep the dependency optional.
        try:
            cohere_module = importlib.import_module("cohere")
        except ImportError as exc:
            raise ProviderRequestError(
                "Cohere SDK is not installed; install with "
                "'pip install cohere'"
            ) from exc

        self.client: cohere.Client = cohere_module.Client(self.api_key)

        # Initialize base class (sets self.model, retry, metrics, etc.)
        # *args removed: positional varargs after keyword-only params cause
        # W1113 and make super().__init__ call order ambiguous.
        super().__init__(provider_name="cohere", model=model, **kwargs)

        # RAG configuration
        self.rag_enabled: bool = rag_enabled
        self.embedding_model: str = embedding_model or "embed-english-v2.0"
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.vector_store_backend: str = vector_store_backend

        # Simple in-memory vector store
        self._vectors: List[List[float]] = []
        self._docs: List[Dict[str, Any]] = []

    # -------------------------
    # Chunking utilities
    # -------------------------
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk *text* into overlapping character windows.

        Args:
            text: Source text to split.

        Returns:
            List of text chunks; empty list when *text* is empty.
        """
        if not text:
            return []

        if self.chunk_size <= 0:
            return [text]

        chunks: List[str] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= text_length:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    # -------------------------
    # Embedding utilities
    # -------------------------
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Embed *texts* via the Cohere Embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors, one per input string.

        Raises:
            ProviderRequestError: On API failure or unexpected response shape.
        """
        if not texts:
            return []

        try:
            resp = self.client.embed(model=self.embedding_model, texts=texts)
            if not hasattr(resp, "embeddings"):
                raise ProviderRequestError(
                    "Cohere embed response missing embeddings field"
                )
            embeddings: List[List[float]] = (  # type: ignore[attr-defined]
                resp.embeddings
            )
            if not isinstance(embeddings, list) or len(embeddings) != len(
                texts
            ):
                raise ProviderRequestError("Invalid embeddings returned")
            return embeddings
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise ProviderRequestError(
                f"Cohere embedding request failed: {exc}"
            ) from exc

    # -------------------------
    # Vector store (in-memory)
    # -------------------------
    def upsert_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Chunk, embed, and store *texts* in the in-memory vector store.

        Args:
            texts: Source documents to index.
            metadatas: Optional per-document metadata dicts.
        """
        if not texts:
            return

        chunk_texts: List[str] = []
        chunk_meta: List[Dict[str, Any]] = []
        for doc_idx, doc_text in enumerate(texts):
            doc_meta = (
                metadatas[doc_idx]
                if metadatas and doc_idx < len(metadatas)
                else None
            )
            for chunk_idx, chunk in enumerate(self._chunk_text(doc_text)):
                meta: Dict[str, Any] = {
                    "doc_index": doc_idx,
                    "chunk_index": chunk_idx,
                }
                if doc_meta:
                    meta["metadata"] = doc_meta
                chunk_texts.append(chunk)
                chunk_meta.append(meta)

        if not chunk_texts:
            return

        for vec, txt, meta in zip(
            self._create_embeddings(chunk_texts), chunk_texts, chunk_meta
        ):
            self._vectors.append(vec)
            self._docs.append({"text": txt, "meta": meta})

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Return the cosine similarity between vectors *a* and *b*.

        Args:
            a: First float vector.
            b: Second float vector.

        Returns:
            Cosine similarity in ``[-1, 1]``, or ``0.0`` for zero vectors.
        """
        if not a or not b:
            return 0.0
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query_documents(
        self, query: str, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve the *top_k* most relevant chunks for *query*.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(doc_record, score)`` tuples sorted by descending score.
        """
        if not query or not self._vectors or not self._docs:
            return []

        query_vecs = self._create_embeddings([query])
        if not query_vecs:
            return []
        qvec = query_vecs[0]

        scored: List[Tuple[int, float]] = sorted(
            (
                (idx, self._cosine_similarity(qvec, vec))
                for idx, vec in enumerate(self._vectors)
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        return [
            (self._docs[idx], float(score)) for idx, score in scored[:top_k]
        ]

    # -------------------------
    # Override send behavior for RAG
    # -------------------------
    def _send_impl(self, prompt: str) -> str:
        """Send *prompt* to Cohere, optionally prepending RAG context.

        Args:
            prompt: Validated prompt string.

        Returns:
            Response text from Cohere.

        Raises:
            ProviderRequestError: On Cohere API failure or empty response.
        """
        try:
            final_prompt = prompt
            if self.rag_enabled and self._vectors:
                retrieved = self.query_documents(prompt, top_k=3)
                if retrieved:
                    context = "\n\n---\n\n".join(
                        f"[score={score:.4f}] {doc.get('text', '')}"
                        for doc, score in retrieved
                    )
                    final_prompt = f"Context:\n{context}\n\nUser: {prompt}"

            response = self.client.chat(
                model=self.model, message=final_prompt
            )
            if not response:
                raise ProviderRequestError("Cohere returned empty response")
            text: Optional[str] = getattr(response, "text", None)
            if text is None:
                text = str(response)
            if not text:
                raise ProviderRequestError("Cohere returned empty text")
            return text.strip()

        except ProviderRequestError:
            raise
        except Exception as exc:
            raise ProviderRequestError(
                f"Cohere request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """Perform a lightweight Cohere connectivity test.

        If RAG is enabled, the embedding endpoint is also verified.

        Returns:
            ``True`` if all tested endpoints respond correctly.
        """
        try:
            resp = self.client.chat(model=self.model, message="ping")
            if not (resp and getattr(resp, "text", None)):
                return False
            if self.rag_enabled:
                emb = self._create_embeddings(["ping"])
                return bool(emb and isinstance(emb[0], list))
            return True
        except Exception:  # pylint: disable=broad-exception-caught
            return False
