"""
Cohere provider implementation for ai_cli with Advanced RAG support.

This module integrates Cohere large language models into the ai_cli
provider framework using the official Cohere Python SDK and adds
Advanced Retrieval-Augmented Generation (RAG) features:

- Text chunking (configurable chunk size and overlap)
- Embedding generation via Cohere Embeddings API
- In-memory vector DB (optional backends can be added)
- Vector similarity querying (cosine similarity)
- Automatic context retrieval and augmentation for chat requests

Environment Variables
---------------------
COHERE_API_KEY
    API key used to authenticate with Cohere API.

Example
-------
export COHERE_API_KEY="your_api_key"

Usage
-----
provider = CohereProvider(
    model="command-r",
    rag_enabled=True,                    # enable RAG
    embed_model="embed-english-v2.0",    # embedding model name
    chunk_size=500,
    chunk_overlap=50,
)

# add documents to vector DB (they will be chunked and embedded)
provider.upsert_documents(["Long document text ..."], metadatas=[{"title": "Doc1"}])

# send a query that uses retrieved context
response = provider.send("Explain Kubernetes operators")
print(response)
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
    AI provider implementation for Cohere models with RAG support.

    Parameters
    ----------
    model : Optional[str]
        Cohere model name for chat/completions (default "command-r").
    api_key : Optional[str]
        Cohere API key; if omitted, will use COHERE_API_KEY environment var.
    rag_enabled : bool
        Whether Retrieval Augmented Generation is enabled (default False).
    embed_model : Optional[str]
        Cohere embedding model name. Defaults to "embed-english-v2.0".
    chunk_size : int
        Character-based chunk size for long documents (default 500).
    chunk_overlap : int
        Character overlap between consecutive chunks (default 50).
    vector_store_backend : str
        Which backend to use for the vector DB. Currently "memory" (default).
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
        super().__init__(model=model or "command-r", api_key=api_key, *args, **kwargs)

        if not self.api_key:
            raise ProviderRequestError("COHERE_API_KEY environment variable is not set")

        # Import Cohere SDK at runtime to keep dependency optional
        try:
            cohere = importlib.import_module("cohere")
        except Exception as exc:
            raise ProviderRequestError(
                "Cohere SDK is not installed; install it with 'pip install cohere'"
            ) from exc

        self.client: "cohere.Client" = cohere.Client(self.api_key)

        # RAG configuration
        self.rag_enabled = bool(rag_enabled)
        self.embed_model = embed_model or "embed-english-v2.0"
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.vector_store_backend = vector_store_backend

        # Simple in-memory vector store: list of vectors and corresponding docs
        self._vectors: List[List[float]] = []
        self._docs: List[Dict[str, Any]] = []

    # -------------------------
    # Chunking utilities
    # -------------------------
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into overlapping character windows.

        This is a simple, dependency-free chunker. It aims to create chunks
        roughly chunk_size characters long with chunk_overlap characters of overlap.

        Returns a list of text chunks (strings).
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
        Create embeddings for a list of texts using Cohere Embeddings API.

        Returns a list of float vectors (one per text). Raises ProviderRequestError on failure.
        """
        if not texts:
            return []

        try:
            # Cohere embed API: client.embed(model=..., texts=[...])
            resp = self.client.embed(model=self.embed_model, texts=texts)
            if not hasattr(resp, "embeddings"):
                raise ProviderRequestError("Cohere embed response missing embeddings field")
            embeddings = resp.embeddings  # type: ignore
            # Basic validation
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ProviderRequestError("Invalid embeddings returned by Cohere")
            return embeddings  # type: ignore
        except Exception as exc:
            raise ProviderRequestError(f"Cohere embedding request failed: {exc}") from exc

    # -------------------------
    # Vector store (in-memory)
    # -------------------------
    def upsert_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add or update documents into the vector store. Documents will be chunked
        and each chunk embedded and stored.

        texts: list of full document strings
        metadatas: optional list of metadata dicts per document (same len as texts)
        """
        if not texts:
            return

        metadatas = metadatas or [None] * len(texts)

        # Prepare chunks with metadata references
        chunk_texts: List[str] = []
        chunk_meta: List[Dict[str, Any]] = []
        for doc_idx, doc_text in enumerate(texts):
            doc_meta = metadatas[doc_idx] if doc_idx < len(metadatas) else None
            chunks = self._chunk_text(doc_text)
            for i, chunk in enumerate(chunks):
                meta = {"doc_index": doc_idx, "chunk_index": i}
                if doc_meta:
                    meta["metadata"] = doc_meta
                chunk_texts.append(chunk)
                chunk_meta.append(meta)

        if not chunk_texts:
            return

        # Create embeddings for chunks
        embeddings = self._create_embeddings(chunk_texts)

        # Upsert into in-memory vectors and docs list
        for vec, txt, meta in zip(embeddings, chunk_texts, chunk_meta):
            doc_record = {
                "text": txt,
                "meta": meta,
            }
            self._vectors.append(vec)
            self._docs.append(doc_record)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Compute cosine similarity between two vectors. Uses math only to avoid heavy deps.
        """
        if not a or not b:
            return 0.0
        # dot product and norms
        dot = 0.0
        na = 0.0
        nb = 0.0
        # assume same length
        for ai, bi in zip(a, b):
            dot += ai * bi
            na += ai * ai
            nb += bi * bi
        if na == 0 or nb == 0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def query_documents(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Query the vector store using the query string. Returns top_k (doc_record, score) tuples.
        """
        if not query:
            return []

        if not self._vectors or not self._docs:
            return []

        query_vecs = self._create_embeddings([query])
        if not query_vecs:
            return []
        qvec = query_vecs[0]

        # score all documents
        scored: List[Tuple[int, float]] = []
        for idx, vec in enumerate(self._vectors):
            score = self._cosine_similarity(qvec, vec)
            scored.append((idx, score))

        # sort by score descending
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

        If RAG is enabled and the vector store has documents, a retrieval
        step is performed to prepend relevant context to the prompt.
        """
        try:
            final_prompt = prompt
            if self.rag_enabled and self._vectors:
                # retrieve top-k docs
                retrieved = self.query_documents(prompt, top_k=3)
                if retrieved:
                    context_pieces = []
                    for doc, score in retrieved:
                        # include a short header and the chunk text
                        context_pieces.append(f"[score={score:.4f}] {doc.get('text','')}")
                    context = "\n\n---\n\n".join(context_pieces)
                    # Build augmented prompt (simple concatenation; can be adapted)
                    final_prompt = f"Context:\n{context}\n\nUser: {prompt}"

            response = self.client.chat(model=self.model, message=final_prompt)
            if not response:
                raise ProviderRequestError("Cohere returned empty response")
            # some SDK versions return .text, others .message or .output - prefer .text
            text = getattr(response, "text", None)
            if text is None:
                # fallback to string representation
                text = str(response)
            if not text:
                raise ProviderRequestError("Cohere returned empty text response")
            return text.strip()

        except Exception as exc:
            raise ProviderRequestError(f"Cohere request failed: {exc}") from exc

    def health_check(self) -> bool:
        """
        Perform lightweight Cohere connectivity test.

        If RAG is enabled, also verify embedding endpoint by creating a tiny embedding.
        """
        try:
            # quick chat ping
            resp = self.client.chat(model=self.model, message="ping")
            chat_ok = bool(resp and getattr(resp, "text", None))
            if not chat_ok:
                return False
            if self.rag_enabled:
                # lightweight embed test
                emb = self._create_embeddings(["ping"])
                return bool(emb and isinstance(emb[0], list))
            return True
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "cohere"