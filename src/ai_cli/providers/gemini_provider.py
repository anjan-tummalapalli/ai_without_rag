"""
Optimized Gemini provider implementation for ai_cli with Advanced RAG support.

Optimizations:
- InMemoryVectorDB precomputes norms to avoid repeated sqrt during queries
  and uses heapq.nlargest for top_k.
- chunk_text uses a safe stepping strategy and avoids unnecessary checks.
- _create_embeddings normalizes access to SDK response and supports
  single/multiple inputs robustly.
- Minor error handling and small performance/clarity improvements across
  methods.
"""
# pylint: disable=too-few-public-methods  # shim stubs inside except block
from __future__ import annotations

import heapq
import math
import os
import warnings
from typing import Any

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider

warnings.filterwarnings("ignore", category=FutureWarning)


# Minimal shim used when neither Google Generative AI SDK is installed.
# Defined at module level so pylint suppression works correctly.
class _GenaiShim:  # pylint: disable=too-few-public-methods
    """Shim that raises ProviderRequestError when no Google SDK is found."""

    def configure(self, *_args, **_kwargs) -> None:
        """Raise immediately — SDK not installed."""
        raise ProviderRequestError(
            "Google Generative AI SDK is not installed; "
            "install 'google-generativeai' or 'google-genai'."
        )

    class GenerativeModel:  # pylint: disable=too-few-public-methods
        """Stub GenerativeModel that raises when SDK is missing."""

        def __init__(self, *_args, **_kwargs) -> None:
            """Raise immediately — SDK not installed."""
            raise ProviderRequestError(
                "Google Generative AI SDK is not installed; "
                "cannot create model."
            )

    class Client:  # pylint: disable=too-few-public-methods
        """Stub Client that raises when SDK is missing."""

        def __init__(self, *_args, **_kwargs) -> None:
            """Raise immediately — SDK not installed."""
            raise ProviderRequestError(
                "Google Generative AI SDK is not installed; "
                "cannot create client."
            )

        class models:  # pylint: disable=invalid-name,too-few-public-methods
            """Stub models namespace."""

            @staticmethod
            def generate_content(*_args, **_kwargs) -> None:
                """Raise immediately — SDK not installed."""
                raise ProviderRequestError(
                    "Google Generative AI SDK is not installed; "
                    "cannot generate content."
                )


try:
    import google.generativeai as genai  # type: ignore  # Legacy SDK
    _GENAI_LEGACY = True
except ImportError:
    try:
        from google import genai  # type: ignore  # New SDK
        _GENAI_LEGACY = False
    except ImportError:
        genai = _GenaiShim()  # type: ignore[assignment]
        _GENAI_LEGACY = False


class InMemoryVectorDB:
    """Minimal in-memory vector DB for storing embeddings.

    Stored items are dicts with keys:
        - id: unique identifier (str)
        - vector: list[float]
        - norm: precomputed L2 norm (float)
        - metadata: dict
        - text: original chunk text (str)
    """

    def __init__(self) -> None:
        """Initialise an empty vector store."""
        self._items: list[dict[str, Any]] = []

    def upsert(self, items: list[dict[str, Any]]) -> None:
        """Insert or replace items by id, precomputing each vector's norm.

        Args:
            items: Sequence of dicts with at least ``id`` and ``vector`` keys.
        """
        ids = {it["id"] for it in items}
        self._items = [it for it in self._items if it["id"] not in ids]
        for it in items:
            vec = it.get("vector", [])
            norm = math.sqrt(sum(x * x for x in vec)) if vec else 0.0
            self._items.append(
                {
                    "id": it["id"],
                    "vector": list(vec),
                    "norm": norm,
                    "metadata": it.get("metadata", {}),
                    "text": it.get("text", ""),
                }
            )

    @staticmethod
    def _cosine_similarity_with_norms(
        a: list[float],
        a_norm: float,
        b: list[float],
        b_norm: float,
    ) -> float:
        """Return cosine similarity using precomputed norms.

        Args:
            a: First vector.
            a_norm: L2 norm of *a*.
            b: Second vector.
            b_norm: L2 norm of *b*.

        Returns:
            Cosine similarity, or ``0.0`` when either norm is zero.
        """
        if a_norm == 0.0 or b_norm == 0.0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (a_norm * b_norm)

    def query(
        self, vector: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Return *top_k* items most similar to *vector*.

        Args:
            vector: Query vector.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts with ``id``, ``score``, ``metadata``,
            and ``text`` keys, sorted by descending score.
        """
        if not self._items:
            return []
        vec = list(vector)
        vec_norm = math.sqrt(sum(x * x for x in vec)) if vec else 0.0

        scored = [
            {
                "id": it["id"],
                "score": self._cosine_similarity_with_norms(
                    vec, vec_norm, it["vector"], it["norm"]
                ),
                "metadata": it.get("metadata", {}),
                "text": it.get("text", ""),
            }
            for it in self._items
        ]

        if top_k >= len(scored):
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored
        return heapq.nlargest(top_k, scored, key=lambda x: x["score"])


class GeminiProvider(AIProvider):
    """AI provider for Google Gemini models with optional RAG support."""

    provider_name = "gemini"

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        embedding_model: str | None = None,
        vector_db_client: Any | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs,
    ) -> None:
        """Initialise GeminiProvider.

        Args:
            model: Gemini model name; defaults to ``"gemini-1.5-flash"``.
            api_key: Google API key; falls back to ``GEMINI_API_KEY`` env var.
            embedding_model: Optional embedding model name for RAG.
            vector_db_client: Optional custom vector store; uses
                InMemoryVectorDB when not provided.
            chunk_size: Character chunk size for RAG splitting.
            chunk_overlap: Character overlap between consecutive chunks.
            **kwargs: Forwarded to AIProvider.__init__.
        """
        super().__init__(
            model=model or "gemini-1.5-flash",
            api_key=api_key,
            **kwargs,
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderRequestError(
                "GEMINI_API_KEY environment variable is not set"
            )

        genai.configure(api_key=self.api_key)

        # Detect which SDK generation is available and create the client.
        try:
            self.client = genai.GenerativeModel(self.model)
            self._use_new_api = False
        except Exception:  # pylint: disable=broad-exception-caught
            self.client = genai.Client()
            self._use_new_api = True

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        self.embedding_model = embedding_model
        self.vector_db = vector_db_client or InMemoryVectorDB()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -----------------------
    # Core send & health
    # -----------------------
    def _send_impl(self, prompt: str) -> str:
        """Send *prompt* to Gemini and return the response text.

        Args:
            prompt: Validated prompt string.

        Returns:
            Stripped response text.

        Raises:
            ProviderRequestError: On API failure or empty/missing response.
        """
        try:
            if self._use_new_api:
                response = self.client.models.generate_content(
                    model=self.model, contents=prompt
                )
            else:
                response = self.client.generate_content(prompt)
        except Exception as exc:
            raise ProviderRequestError(
                f"Gemini request failed: {exc}"
            ) from exc

        if not response:
            raise ProviderRequestError("Gemini returned empty response")
        text = getattr(response, "text", None) or (
            response.get("text") if isinstance(response, dict) else None
        )
        if not text:
            raise ProviderRequestError("Gemini response missing text field")
        return text.strip()

    def health_check(self) -> bool:
        """Perform a lightweight Gemini connectivity test.

        Returns:
            ``True`` if the model responds with non-empty text.
        """
        try:
            if self._use_new_api:
                response = self.client.models.generate_content(
                    model=self.model, contents="ping"
                )
            else:
                response = self.client.generate_content("ping")
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        text = getattr(response, "text", None) or (
            response.get("text") if isinstance(response, dict) else None
        )
        return bool(text)

    # -----------------------
    # Chunking utilities
    # -----------------------
    def chunk_text(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None,
    ) -> list[str]:
        """Split *text* into overlapping character-window chunks.

        Args:
            text: Source text to split.
            chunk_size: Override for ``self.chunk_size``.
            overlap: Override for ``self.chunk_overlap``.

        Returns:
            List of text chunks; empty when *text* is empty.
        """
        chunk_size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = overlap if overlap is not None else self.chunk_overlap

        # Ensure forward progress on every iteration.
        step = max(chunk_size - overlap, 1)
        text_len = len(text)
        if text_len == 0:
            return []

        chunks: list[str] = []
        for start in range(0, text_len, step):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= text_len:
                break
        return chunks

    # -----------------------
    # Embeddings utilities
    # -----------------------
    def _create_embeddings(self, inputs: list[str]) -> list[list[float]]:
        """Return embedding vectors for *inputs* via the Gemini embedding API.

        Args:
            inputs: Texts to embed.

        Returns:
            List of float vectors, one per input string.

        Raises:
            ProviderRequestError: On API unavailability, empty data, or
                missing embedding vectors in the response.        """
        if not inputs:
            return []

        if not hasattr(genai, "embeddings") or not hasattr(
            genai.embeddings, "create"
        ):
            raise ProviderRequestError(
                "Embedding API not available in google-generativeai SDK"
            )

        extra: dict[str, Any] = {}
        if self.embedding_model:
            extra["model"] = self.embedding_model

        payload = list(inputs)

        try:
            result = genai.embeddings.create(input=payload, **extra)
            data = getattr(result, "data", None) or (
                result.get("data") if isinstance(result, dict) else None
            )
            if not data:
                raise ProviderRequestError("Embedding API returned no data")

            vectors: list[list[float]] = []
            for item in data:
                emb = (
                    item.get("embedding")
                    if isinstance(item, dict)
                    else getattr(item, "embedding", None)
                )
                if emb is None:
                    raise ProviderRequestError(
                        "Embedding item missing 'embedding' vector"
                    )
                vectors.append(list(emb))
            return vectors
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise ProviderRequestError(
                f"Failed to create embeddings: {exc}"
            ) from exc

    # -----------------------
    # Indexing and querying
    # -----------------------
    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any | None] = None,
    ) -> None:
        """Chunk *text*, embed each chunk, and upsert into the vector store.

        Args:
            doc_id: Unique document identifier (used to namespace chunk IDs).
            text: Source document text.
            metadata: Optional metadata dict attached to every chunk.

        Raises:
            ProviderRequestError: On embedding failure or vector DB error.
        """
        chunks = self.chunk_text(text)
        if not chunks:
            return

        vectors = self._create_embeddings(chunks)
        if len(vectors) != len(chunks):
            raise ProviderRequestError(
                "Embedding count does not match chunk count"
            )

        chunk_items: list[dict[str, Any]] = [
            {
                "id": f"{doc_id}::chunk::{i}",
                "vector": vec,
                "metadata": metadata or {},
                "text": chunk,
            }
            for i, (chunk, vec) in enumerate(zip(chunks, vectors, strict=True))
        ]

        try:
            self.vector_db.upsert(chunk_items)
        except Exception as exc:
            raise ProviderRequestError(
                f"Failed to upsert vectors to vector DB: {exc}"
            ) from exc

    def query_vector_db(
        self, query: str, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Embed *query* and return the *top_k* nearest chunks.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts from the vector store.

        Raises:
            ProviderRequestError: On embedding or vector DB failure.
        """
        try:
            q_vecs = self._create_embeddings([query])
            if not q_vecs or len(q_vecs) != 1:
                raise ProviderRequestError(
                    "Failed to create embedding for query"
                )
            q_vec = q_vecs[0]
        except ProviderRequestError:
            raise
        except Exception as exc:
            raise ProviderRequestError(
                f"Query embedding failed: {exc}"
            ) from exc

        try:
            return self.vector_db.query(q_vec, top_k=top_k)
        except Exception as exc:
            raise ProviderRequestError(
                f"Vector DB query failed: {exc}"
            ) from exc

    def retrieve_relevant_context(
        self,
        query: str,
        top_k: int = 3,
        joiner: str = "\n---\n",
    ) -> str:
        """Return joined text from the *top_k* most relevant stored chunks.

        Args:
            query: Natural-language query string.
            top_k: Number of chunks to retrieve.
            joiner: Separator between retrieved text snippets.

        Returns:
            Concatenated context string, or ``""`` when nothing is found.
        """
        results = self.query_vector_db(query, top_k=top_k)
        if not results:
            return ""
        texts = [res.get("text", "") for res in results if res.get("text")]
        return joiner.join(texts)

    # -----------------------
    # High-level RAG send
    # -----------------------
    def send_with_rag(
        self,
        prompt: str,
        top_k: int = 3,
        prepend_context: bool = True,
        context_prefix: str | None = None,
    ) -> str:
        """Retrieve context from the vector store and send augmented prompt.

        Augments *prompt* with retrieved context before calling the model.

        Args:
            prompt: User prompt to augment and send.
            top_k: Number of context chunks to retrieve.
            prepend_context: When ``True``, context appears before the prompt;
                otherwise it appears after.
            context_prefix: Label line before the context block; defaults to
                ``"Context (retrieved):"``.

        Returns:
            Model response string.

        Raises:
            ProviderRequestError: When no embedding model is configured.
        """
        if not self.embedding_model:
            raise ProviderRequestError(
                "Embedding model not configured; cannot perform RAG. "
                "Provide `embedding_model` when creating the provider."
            )

        context = self.retrieve_relevant_context(prompt, top_k=top_k)
        if context:
            prefix = (
                context_prefix
                if context_prefix is not None
                else "Context (retrieved):"
            )
            if prepend_context:
                combined = f"{prefix}\n{context}\n\n{prompt}"
            else:
                combined = f"{prompt}\n\n{prefix}\n{context}"
        else:
            combined = prompt

        return self._send_impl(combined)
    
    def send(self, prompt: str, **kwargs) -> str:
        return self._send_impl(prompt)
