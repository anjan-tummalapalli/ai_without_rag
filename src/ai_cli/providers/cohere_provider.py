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

import os
import importlib
from typing import Optional, TYPE_CHECKING, List, Dict, Any, Tuple

if TYPE_CHECKING:
    import cohere  # type: ignore

import math

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider


class CohereProvider(AIProvider):
    """
    Cohere AI provider with optional RAG.

    Parameters
    ----------
    model: Cohere model name, default "command-r".
    api_key: Cohere API key or use COHERE_API_KEY env var.
    rag_enabled: enable Retrieval Augmented Generation.
    embed_model: embedding model, default "embed-english-v2.0".
    chunk_size: char chunk size, default 500.
    chunk_overlap: char overlap between chunks, default 50.
    vector_store_backend: backend for vector DB, default "memory".
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        rag_enabled: bool = False,
        embed_model: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_backend: str = "memory",
        *args,
        **kwargs,
    ) -> None:
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ProviderRequestError(
                "COHERE_API_KEY environment variable is not set"
            )

        # Import Cohere SDK at runtime to keep dependency optional
        try:
            cohere = importlib.import_module("cohere")
        except Exception as exc:
            raise ProviderRequestError(
                "Cohere SDK is not installed; install with "
                "'pip install cohere'"
            ) from exc

        self.client: "cohere.Client" = cohere.Client(self.api_key)

        # Initialize base class (sets self.model, retry, metrics, etc.)
        super().__init__(provider_name="cohere", model=model, *args, **kwargs)

        # RAG configuration
        self.rag_enabled = rag_enabled
        self.embed_model = embed_model or "embed-english-v2.0"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_backend = vector_store_backend

        # Simple in-memory vector store
        self._vectors: List[List[float]] = []
        self._docs: List[Dict[str, Any]] = []

    # -------------------------
    # Chunking utilities
    # -------------------------
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping character windows.

        Returns a list of text chunks.
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
            chunk = text[start:end]
            chunks.append(chunk)
            if end >= text_length:
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

    # -------------------------
    # Embedding utilities
    # -------------------------
    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts via Cohere Embeddings API.

        Returns list of float vectors, one per input text.
        """
        if not texts:
            return []

        try:
            resp = self.client.embed(model=self.embed_model, texts=texts)
            if not hasattr(resp, "embeddings"):
                raise ProviderRequestError(
                    "Cohere embed response missing embeddings field"
                )
            embeddings = resp.embeddings  # type: ignore
            if not isinstance(embeddings, list) or len(embeddings) != len(
                texts
            ):
                raise ProviderRequestError("Invalid embeddings returned")
            return embeddings  # type: ignore
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
        """
        Add or update documents into the vector store. Documents are chunked
        and each chunk embedded and stored.
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
            chunks = self._chunk_text(doc_text)
            for i, chunk in enumerate(chunks):
                meta: Dict[str, Any] = {
                    "doc_index": doc_idx,
                    "chunk_index": i,
                }
                if doc_meta:
                    meta["metadata"] = doc_meta
                chunk_texts.append(chunk)
                chunk_meta.append(meta)

        if not chunk_texts:
            return

        embeddings = self._create_embeddings(chunk_texts)

        for vec, txt, meta in zip(embeddings, chunk_texts, chunk_meta):
            doc_record = {"text": txt, "meta": meta}
            self._vectors.append(vec)
            self._docs.append(doc_record)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        if not a or not b:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for ai, bi in zip(a, b):
            dot += ai * bi
            na += ai * ai
            nb += bi * bi
        if na == 0 or nb == 0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def query_documents(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Query the vector store using the query string. Returns top_k tuples
        of (doc_record, score).
        """
        if not query:
            return []

        if not self._vectors or not self._docs:
            return []

        query_vecs = self._create_embeddings([query])
        if not query_vecs:
            return []
        qvec = query_vecs[0]

        scored: List[Tuple[int, float]] = []
        for idx, vec in enumerate(self._vectors):
            score = self._cosine_similarity(qvec, vec)
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in scored[:top_k]:
            results.append((self._docs[idx], float(score)))
        return results

    # -------------------------
    # Override send behavior for RAG
    # -------------------------
    def _send_impl(self, prompt: str) -> str:
        """
        Send prompt to Cohere model and return response.

        If RAG is enabled and there are docs, perform retrieval and
        prepend context to the prompt.
        """
        try:
            final_prompt = prompt
            if self.rag_enabled and self._vectors:
                retrieved = self.query_documents(prompt, top_k=3)
                if retrieved:
                    context_pieces = []
                    for doc, score in retrieved:
                        context_pieces.append(
                            f"[score={score:.4f}] {doc.get('text','')}"
                        )
                    context = "\n\n---\n\n".join(context_pieces)
                    final_prompt = (
                        f"Context:\n{context}\n\nUser: {prompt}"
                    )

            response = self.client.chat(
                model=self.model, message=final_prompt
            )
            if not response:
                raise ProviderRequestError("Cohere returned empty response")
            text = getattr(response, "text", None)
            if text is None:
                text = str(response)
            if not text:
                raise ProviderRequestError("Cohere returned empty text")
            return text.strip()

        except Exception as exc:
            raise ProviderRequestError(
                f"Cohere request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Perform a lightweight Cohere connectivity test.

        If RAG is enabled, also test the embedding endpoint.
        """
        try:
            resp = self.client.chat(model=self.model, message="ping")
            chat_ok = bool(resp and getattr(resp, "text", None))
            if not chat_ok:
                return False
            if self.rag_enabled:
                emb = self._create_embeddings(["ping"])
                return bool(emb and isinstance(emb[0], list))
            return True
        except Exception:
            return False
