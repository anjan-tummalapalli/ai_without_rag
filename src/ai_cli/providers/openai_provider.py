"""
OpenAI ChatGPT provider implementation for ai_cli with Advanced RAG support.

This module integrates OpenAI ChatGPT models into the ai_cli provider
framework using the official OpenAI Python SDK and adds an optional
lightweight Retrieval-Augmented Generation (RAG) capability:

Features
--------
- Chat completions (same as before)
- Embeddings (OpenAI Embeddings API)
- Simple chunking (token/word-based) with overlap
- In-memory vector store (NumPy-based) with cosine-similarity nearest-neighbor search
- RAG query helper: retrieve top-k relevant chunks and call chat completion with context
- Save/load of in-memory vector store to disk (NumPy .npz)

Environment Variables
---------------------
OPENAI_API_KEY
    API key used to authenticate with OpenAI API.

Optional Dependencies
---------------------
- numpy
    Used for efficient vector math. Install with:
    pip install numpy

Embedding model recommendation
------------------------------
- text-embedding-3-small (or change to preferred embedding model)

Example
-------
export OPENAI_API_KEY="your_api_key"

provider = OpenAIProvider(model="gpt-5.5")
# Build vector store from documents
provider.build_vector_store([
    {"id": "doc1", "text": "Kubernetes is an open-source container orchestration system."},
    {"id": "doc2", "text": "Operators extend Kubernetes with custom controllers..."},
])
# Ask with RAG
answer = provider.answer_with_rag("What is a Kubernetes operator?", top_k=3)
print(answer)
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any, Tuple

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - fall back not expected in normal use
    np = None  # type: ignore

from openai import OpenAI

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider


class OpenAIProvider(AIProvider):
    """
    AI provider implementation for OpenAI ChatGPT models with RAG utilities.

    New RAG-related methods:
    - chunk_text(text, chunk_size=500, overlap=50)
    - embed_chunks(chunks, embedding_model="text-embedding-3-small")
    - build_vector_store(documents, chunk_size=500, overlap=50)
    - query_vector_store(query_embedding, top_k=5)
    - answer_with_rag(query, top_k=5, prompt_template=None)

    The in-memory vector store keeps:
    - self._vectors : np.ndarray shape (N, D)
    - self._metadatas : List[Dict] with keys: source_id, chunk_index, text
    """

    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model or "gpt-5.5",
            api_key=api_key,
            *args,
            **kwargs,
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ProviderRequestError(
                "OPENAI_API_KEY environment variable is not set"
            )

        # Create OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

        # Lightweight in-memory vector store
        self._vectors = None  # type: Optional[Any]  # numpy array when built
        self._metadatas: List[Dict[str, Any]] = []

    def _send_impl(self, prompt: str) -> str:
        """
        Send prompt to OpenAI ChatGPT model and return text content.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.7,
            )

            if not getattr(response, "choices", None):
                raise ProviderRequestError(
                    "OpenAI returned no completion choices"
                )

            message = response.choices[0].message

            if not message or not getattr(message, "content", None):
                raise ProviderRequestError(
                    "OpenAI returned empty response content"
                )

            return message.content.strip()

        except Exception as exc:
            raise ProviderRequestError(
                f"OpenAI request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Perform lightweight OpenAI connectivity test.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "ping",
                    }
                ],
                max_tokens=5,
            )

            return bool(getattr(response, "choices", None))

        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "openai"

    # ----------------------------
    # RAG utility methods
    # ----------------------------
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk text into overlapping segments by words.

        Parameters
        ----------
        text : str
            Source text to chunk.
        chunk_size : int
            Approximate number of words per chunk.
        overlap : int
            Number of overlapping words between chunks.

        Returns
        -------
        List[str]
            List of text chunks.
        """
        words = text.split()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            overlap = 0

        chunks: List[str] = []
        start = 0
        n = len(words)
        while start < n:
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            if end >= n:
                break
            start = end - overlap

        return chunks

    def embed_chunks(
        self,
        chunks: List[str],
        embedding_model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Create embeddings for a list of chunks using OpenAI Embeddings API.

        Returns
        -------
        List[List[float]]
            List of vector embeddings (each a list of floats).
        """
        model = embedding_model or self.DEFAULT_EMBEDDING_MODEL

        try:
            # The OpenAI Python SDK returns .data with embeddings in .embedding
            response = self.client.embeddings.create(
                model=model,
                input=chunks,
            )
            embeddings = []
            for item in response.data:
                # Some SDKs return item.embedding, others item["embedding"]
                emb = getattr(item, "embedding", None) or item.get("embedding")  # type: ignore
                if emb is None:
                    raise ProviderRequestError("Embedding response missing embedding field")
                embeddings.append(list(emb))
            return embeddings

        except Exception as exc:
            raise ProviderRequestError(f"Embedding request failed: {exc}") from exc

    def build_vector_store(
        self,
        documents: List[Dict[str, str]],
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_model: Optional[str] = None,
    ) -> None:
        """
        Build or extend the in-memory vector store from a list of documents.

        documents: List of {"id": "<doc_id>", "text": "<document_text>"}
        """
        all_chunks: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = doc.get("id") or doc.get("source") or "unknown"
            text = doc.get("text", "")
            chunks = self.chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for idx, chunk in enumerate(chunks):
                metadatas.append({"source_id": doc_id, "chunk_index": idx, "text": chunk})
                all_chunks.append(chunk)

        if not all_chunks:
            return

        embeddings = self.embed_chunks(all_chunks, embedding_model=embedding_model)

        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")

        vecs = np.array(embeddings, dtype=float)
        # Normalize vectors to unit length for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        if self._vectors is None:
            self._vectors = vecs
            self._metadatas = metadatas
        else:
            # append
            self._vectors = np.vstack([self._vectors, vecs])
            self._metadatas.extend(metadatas)

    def query_vector_store(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Query the in-memory vector store with a query embedding.

        Returns top_k list of (score, metadata).
        Score is cosine similarity in [0,1].
        """
        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")

        if self._vectors is None or len(self._metadatas) == 0:
            return []

        q = np.array(query_embedding, dtype=float)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0
        q = q / q_norm

        # cosine similarity via dot product since vectors are normalized
        scores = (self._vectors @ q).tolist()
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for idx, score in indexed[:top_k]:
            results.append((float(score), self._metadatas[idx]))
        return results

    def answer_with_rag(
        self,
        query: str,
        top_k: int = 5,
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_model: Optional[str] = None,
        prompt_template: Optional[str] = None,
    ) -> str:
        """
        Perform RAG: embed query, retrieve top-k chunks, and ask the chat model
        with retrieved context.

        prompt_template can include placeholders:
            {context} - will be replaced with concatenated retrieved chunk texts
            {query}   - original user query

        If prompt_template is None, a simple default prompt is used.
        """
        # Ensure vector store exists; if empty, no retrieval will be performed
        if self._vectors is None or len(self._metadatas) == 0:
            raise ProviderRequestError("Vector store is empty. Call build_vector_store first.")

        # Embed the query
        try:
            emb_resp = self.client.embeddings.create(
                model=embedding_model or self.DEFAULT_EMBEDDING_MODEL,
                input=[query],
            )
            item = emb_resp.data[0]
            query_emb = getattr(item, "embedding", None) or item.get("embedding")  # type: ignore
            if query_emb is None:
                raise ProviderRequestError("Embedding response missing embedding field")
        except Exception as exc:
            raise ProviderRequestError(f"Embedding request failed: {exc}") from exc

        # Retrieve top_k
        hits = self.query_vector_store(query_emb, top_k=top_k)
        if not hits:
            # fallback to direct send
            return self._send_impl(query)

        contexts = []
        for score, meta in hits:
            contexts.append(f"[{meta.get('source_id')}#chunk{meta.get('chunk_index')}] {meta.get('text')}")

        context_text = "\n\n---\n\n".join(contexts)

        if prompt_template is None:
            prompt = (
                "You are an assistant that answers user queries using the provided context. "
                "If the answer is not contained in the context, say you don't know.\n\n"
                "Context:\n"
                "{context}\n\n"
                "Question:\n"
                "{query}\n\n"
                "Answer concisely:"
            ).format(context=context_text, query=query)
        else:
            prompt = prompt_template.format(context=context_text, query=query)

        return self._send_impl(prompt)

    def save_vector_store(self, path: str) -> None:
        """
        Save the in-memory vector store to a .npz file.

        The file will contain:
            vectors: float32 array (N,D)
            metadatas: list of dicts (saved as object)
        """
        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")
        if self._vectors is None:
            raise ProviderRequestError("No vector store to save")

        try:
            # Use numpy.savez to persist vectors; metadata saved via object array
            np.savez_compressed(path, vectors=self._vectors.astype(np.float32), metadatas=np.array(self._metadatas, dtype=object))
        except Exception as exc:
            raise ProviderRequestError(f"Failed to save vector store: {exc}") from exc

    def load_vector_store(self, path: str) -> None:
        """
        Load vector store from a .npz file saved by save_vector_store.
        """
        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")
        try:
            data = np.load(path, allow_pickle=True)
            vectors = data["vectors"]
            metadatas = data["metadatas"].tolist()
            # Normalize just in case
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
            self._vectors = vectors
            self._metadatas = metadatas
        except Exception as exc:
            raise ProviderRequestError(f"Failed to load vector store: {exc}") from exc