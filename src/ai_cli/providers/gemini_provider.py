"""
Gemini provider implementation for ai_cli with Advanced RAG support.

This module provides integration with Google's Gemini models
using the official google-generativeai SDK and adds an optional
Retrieval-Augmented Generation (RAG) pipeline with:

- Text chunking (configurable size & overlap)
- Embedding creation (using a specified embedding model)
- Vector DB support (pluggable client or built-in in-memory store)
- Vector querying and context retrieval
- Convenience send_with_rag method that injects retrieved context
    into the prompt before sending to Gemini

Environment Variables
---------------------
GEMINI_API_KEY
        API key used to authenticate with Google Gemini API.

Optional parameters for RAG (when creating provider)
- embedding_model : str
        Embedding model name to use (if None, embeddings are attempted
        via the SDK's default or an error is raised if embeddings are unsupported).
- vector_db_client : object
        Optional vector DB client implementing `upsert(items)` and `query(vector, top_k)`
        If not provided, an in-memory vector DB will be used.

Example
-------
export GEMINI_API_KEY="your_api_key"

provider = GeminiProvider(
        model="gemini-1.5-flash",
        embedding_model="embedding-model-name",      # optional
)

# Add documents to the vector DB
provider.index_document(doc_id="doc1", text="Long document text ...")

# Use RAG-enabled send
response = provider.send_with_rag("Explain Kubernetes operators", top_k=3)
print(response)
"""

from __future__ import annotations

import os
import math
from typing import Optional, Sequence, List, Dict, Any, Tuple

from ai_cli.core.exceptions import ProviderRequestError

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    # Minimal shim so static analysis and runtime usage produce clear errors when the SDK is missing.
    class _GenaiShim:
        def configure(self, *args, **kwargs):
            raise ProviderRequestError(
                "google-generativeai SDK is not installed; please install it (pip install google-generativeai)"
            )

        class GenerativeModel:
            def __init__(self, *args, **kwargs):
                raise ProviderRequestError(
                    "google-generativeai SDK is not installed; please install it (pip install google-generativeai)"
                )

        class embeddings:
            @staticmethod
            def create(*args, **kwargs):
                raise ProviderRequestError(
                    "google-generativeai SDK is not installed; cannot create embeddings"
                )

    genai = _GenaiShim()

from ai_cli.providers.base import AIProvider


class InMemoryVectorDB:
        """
        Minimal in-memory vector DB for storing embeddings.

        Stored items: list of dicts with keys:
            - id: unique id (str)
            - vector: List[float]
            - metadata: dict
            - text: original chunk text
        """

        def __init__(self) -> None:
                self._items: List[Dict[str, Any]] = []

        def upsert(self, items: Sequence[Dict[str, Any]]) -> None:
                # Replace items with same id or append otherwise
                ids = {it["id"] for it in items}
                # Remove existing with same ids
                self._items = [it for it in self._items if it["id"] not in ids]
                # Append new items
                self._items.extend(items)

        def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
                # defensive: handle zero vectors
                denom_a = math.sqrt(sum(x * x for x in a))
                denom_b = math.sqrt(sum(x * x for x in b))
                if denom_a == 0 or denom_b == 0:
                        return 0.0
                dot = sum(x * y for x, y in zip(a, b))
                return dot / (denom_a * denom_b)

        def query(self, vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
                # Compute similarity against each stored vector
                scored = []
                for it in self._items:
                        score = self._cosine_similarity(vector, it["vector"])
                        scored.append({"id": it["id"], "score": score, "metadata": it.get("metadata", {}), "text": it.get("text", "")})
                scored.sort(key=lambda x: x["score"], reverse=True)
                return scored[:top_k]


class GeminiProvider(AIProvider):
        """
        AI provider implementation for Google Gemini models with optional RAG.

        See AIProvider for base interface.
        """

        def __init__(
                self,
                model: Optional[str] = None,
                api_key: Optional[str] = None,
                embedding_model: Optional[str] = None,
                vector_db_client: Optional[Any] = None,
                chunk_size: int = 500,
                chunk_overlap: int = 50,
                *args,
                **kwargs,
        ) -> None:
                """
                Initialize Gemini provider with optional RAG components.

                Parameters
                ----------
                model : str | None
                        Gemini model name to use (defaults to 'gemini-1.5-flash').

                api_key : str | None
                        Gemini API key. If not provided, reads from GEMINI_API_KEY.

                embedding_model : str | None
                        Embedding model name to use for embeddings (optional).
                        If provided, embeddings will be requested via the SDK's embedding
                        endpoint. If omitted and RAG methods are used, an error may be raised.

                vector_db_client : object | None
                        Optional vector DB client with methods:
                            - upsert(items: Sequence[dict]) -> None
                            - query(vector: Sequence[float], top_k: int) -> List[dict]
                        If not provided, an in-memory vector DB is used.

                chunk_size : int
                        Chunk size (characters) for text splitting.

                chunk_overlap : int
                        Overlap (characters) between chunks.

                Raises
                ------
                ProviderRequestError
                        If API key is missing.
                """
                super().__init__(model=model or "gemini-1.5-flash", api_key=api_key, *args, **kwargs)
                self.api_key = api_key or os.getenv("GEMINI_API_KEY")
                if not self.api_key:
                        raise ProviderRequestError("GEMINI_API_KEY environment variable is not set")

                # Configure Gemini SDK
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model)

                # RAG-related
                self.embedding_model = embedding_model
                self.vector_db = vector_db_client or InMemoryVectorDB()
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

        # -----------------------
        # Core send & health
        # -----------------------
        def _send_impl(self, prompt: str) -> str:
                """
                Send a prompt to Gemini for generation and return the generated text.
                """
                try:
                        # Try the standard generate_content method
                        response = self.client.generate_content(prompt)
                        if not response:
                                raise ProviderRequestError("Gemini returned empty response")
                        if not hasattr(response, "text"):
                                raise ProviderRequestError("Gemini response missing text field")
                        return response.text.strip()
                except Exception as exc:
                        raise ProviderRequestError(f"Gemini request failed: {exc}") from exc

        def health_check(self) -> bool:
                try:
                        response = self.client.generate_content("ping")
                        return bool(response and getattr(response, "text", None))
                except Exception:
                        return False

        @property
        def provider_name(self) -> str:
                return "gemini"

        # -----------------------
        # Chunking utilities
        # -----------------------
        def chunk_text(self, text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
                """
                Split text into overlapping chunks.

                Parameters
                ----------
                text : str
                chunk_size : int | None
                        If None, uses provider default.
                overlap : int | None
                        If None, uses provider default.

                Returns
                -------
                List[str]
                        List of text chunks.
                """
                if chunk_size is None:
                        chunk_size = self.chunk_size
                if overlap is None:
                        overlap = self.chunk_overlap
                if chunk_size <= 0:
                        raise ValueError("chunk_size must be positive")
                if overlap < 0:
                        raise ValueError("overlap must be non-negative")

                chunks: List[str] = []
                start = 0
                text_len = len(text)
                while start < text_len:
                        end = start + chunk_size
                        chunks.append(text[start:end])
                        if end >= text_len:
                                break
                        start = end - overlap if (end - overlap) > start else end
                return chunks

        # -----------------------
        # Embeddings utilities
        # -----------------------
        def _create_embeddings(self, inputs: Sequence[str]) -> List[List[float]]:
                """
                Create embeddings for a list of strings.

                This method attempts to use the google.generativeai SDK's embeddings endpoint.
                Raise ProviderRequestError on failure or if embeddings are unsupported.
                """
                if not inputs:
                        return []

                # If user provided an embedding_model name, pass it through.
                try:
                        # The SDK's embeddings API may differ between versions. Attempt common call.
                        if hasattr(genai, "embeddings") and hasattr(genai.embeddings, "create"):
                                kwargs = {"model": self.embedding_model} if self.embedding_model else {}
                                result = genai.embeddings.create(input=list(inputs), **kwargs)
                                # Expecting result.data where each item has a 'embedding' field.
                                vectors = []
                                data = getattr(result, "data", None) or result.get("data", None)  # tolerant access
                                if not data:
                                        raise ProviderRequestError("Embedding API returned no data")
                                for item in data:
                                        emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                                        if emb is None:
                                                raise ProviderRequestError("Embedding item missing 'embedding' vector")
                                        vectors.append(list(emb))
                                return vectors
                        else:
                                # If SDK doesn't expose embeddings as expected, raise
                                raise ProviderRequestError("Embedding API not available in google-generativeai SDK")
                except ProviderRequestError:
                        raise
                except Exception as exc:
                        raise ProviderRequestError(f"Failed to create embeddings: {exc}") from exc

        # -----------------------
        # Indexing and querying
        # -----------------------
        def index_document(self, doc_id: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> None:
                """
                Index a document into the vector DB by chunking and embedding.

                Parameters
                ----------
                doc_id : str
                        Unique identifier for the document (used as prefix for chunk ids).
                text : str
                        Full text to index.
                metadata : dict | None
                        Optional metadata to attach to each chunk.
                """
                chunks = self.chunk_text(text)
                if not chunks:
                        return

                # Create chunk ids and metadata per chunk
                chunk_items: List[Dict[str, Any]] = []
                # Generate embeddings in batch
                vectors = self._create_embeddings(chunks)
                if len(vectors) != len(chunks):
                        raise ProviderRequestError("Embedding count does not match chunk count")

                for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                        item = {
                                "id": f"{doc_id}::chunk::{i}",
                                "vector": vec,
                                "metadata": metadata or {},
                                "text": chunk,
                        }
                        chunk_items.append(item)

                # Upsert to vector DB
                try:
                        self.vector_db.upsert(chunk_items)
                except Exception as exc:
                        raise ProviderRequestError(f"Failed to upsert vectors to vector DB: {exc}") from exc

        def query_vector_db(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
                """
                Embed a query and retrieve top_k similar chunks from the vector DB.

                Returns list of dicts with keys: id, score, metadata, text
                """
                try:
                        q_vecs = self._create_embeddings([query])
                        if not q_vecs or len(q_vecs) != 1:
                                raise ProviderRequestError("Failed to create embedding for query")
                        q_vec = q_vecs[0]
                except ProviderRequestError:
                        raise
                except Exception as exc:
                        raise ProviderRequestError(f"Query embedding failed: {exc}") from exc

                try:
                        results = self.vector_db.query(q_vec, top_k=top_k)
                        return results
                except Exception as exc:
                        raise ProviderRequestError(f"Vector DB query failed: {exc}") from exc

        def retrieve_relevant_context(self, query: str, top_k: int = 3, joiner: str = "\n---\n") -> str:
                """
                Retrieve top_k relevant chunks and join them into a context string.
                """
                results = self.query_vector_db(query, top_k=top_k)
                if not results:
                        return ""
                texts = [res.get("text", "") for res in results if res.get("text")]
                return joiner.join(texts)

        # -----------------------
        # High-level RAG send
        # -----------------------
        def send_with_rag(self, prompt: str, top_k: int = 3, prepend_context: bool = True, context_prefix: Optional[str] = None) -> str:
                """
                Send a prompt augmented with retrieved context.

                Parameters
                ----------
                prompt : str
                        Original user prompt.
                top_k : int
                        Number of relevant chunks to fetch.
                prepend_context : bool
                        If True, retrieved context is prepended to the prompt. Otherwise it's appended.
                context_prefix : str | None
                        Optional label inserted before the context block.

                Returns
                -------
                str
                        Generated response from Gemini.
                """
                if not self.embedding_model:
                        raise ProviderRequestError("Embedding model not configured; cannot perform RAG. Provide `embedding_model` when creating the provider.")

                context = self.retrieve_relevant_context(prompt, top_k=top_k)
                if context:
                        prefix = context_prefix if context_prefix is not None else "Context (retrieved):"
                        combined = f"{prefix}\n{context}\n\n{prompt}" if prepend_context else f"{prompt}\n\n{prefix}\n{context}"
                else:
                        combined = prompt

                return self._send_impl(combined)