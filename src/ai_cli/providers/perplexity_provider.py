"""
Perplexity provider implementation with optional Advanced RAG support.

This module integrates Perplexity AI models into the ai_cli provider framework
using Perplexity's OpenAI-compatible API and adds utilities to:

- chunk documents
- embed chunks
- store embeddings in a lightweight in-memory vector DB
- retrieve context and perform RAG-style calls
"""

from __future__ import annotations

from collections.abc import Sequence

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider
from ai_cli.rag.chunker import chunk_text
from ai_cli.rag.embeddings import EmbeddingsProvider
from ai_cli.rag.models import Chunk
from ai_cli.rag.vector_store import InMemoryVectorStore


class PerplexityProvider(AIProvider):
    BASE_URL = "https://api.perplexity.ai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            provider_name="perplexity",
            model=model or "sonar-pro",
            api_key=api_key,
            **kwargs,
        )

        if OpenAI is None:
            raise ProviderRequestError(
            "The 'openai' package is required to use PerplexityProvider; "
            "install it with `pip install openai`"
            )

        # single client initialization
        self.client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)

        # RAG components (lazy-init)
        self.embeddings_provider: EmbeddingsProvider | None = None
        self.vector_store: InMemoryVectorStore | None = None

    def _to_np_array(self, data, dtype=None):
        """
        Convert data to a numpy array; raise a clear error if numpy is not installed.
        dtype may be a numpy dtype object or a string like 'float32'.
        """
        if np is None:
            raise ProviderRequestError("numpy is required for RAG features; please install it (pip install numpy)")
        if dtype is not None:
            return np.asarray(data, dtype=dtype)
        return np.asarray(data)

    def _create_chat_completion(self, prompt: str, temperature: float | None = None) -> str:
        """Wrapper for chat completion calls with centralized error handling.

        Parameters
        ----------
        prompt: str
            The user prompt to send to Perplexity.
        temperature: Optional[float]
            Sampling temperature (0.0‑2.0). If ``None`` the API default is used.
        """
        try:
            # Build request parameters with explicit Any typing to avoid TypedDict union issues
            from typing import Any  # local import to avoid top‑level pollution
            kwargs: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = self.client.chat.completions.create(**kwargs)
            if not getattr(response, "choices", None):
                raise ProviderRequestError("Perplexity returned no choices")
            message = response.choices[0].message
            if not message or not getattr(message, "content", None):
                raise ProviderRequestError("Perplexity returned empty content")
            return message.content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"Perplexity request failed: {exc}") from exc

    def _send_impl(self, prompt: str) -> str:
        return self._create_chat_completion(prompt)

    def health_check(self) -> bool:
        try:
            # lightweight ping
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": "ping"}]
            )
            return bool(getattr(response, "choices", None))
        except Exception:
            return False

    # -------------------------
    # Advanced RAG helpers
    # -------------------------

    def _ensure_rag_components(self, embed_model: str | None = None) -> None:
        if self.embeddings_provider is None:
            self.embeddings_provider = EmbeddingsProvider(model=embed_model) if embed_model else EmbeddingsProvider()
        if self.vector_store is None:
            self.vector_store = InMemoryVectorStore()

    def build_rag_index(
        self,
        documents: Sequence[str],
        chunk_size: int = 500,
        overlap: int = 50,
        embed_model: str | None = None,
        ids_prefix: str | None = None,
    ) -> None:
        """
        Build an in-memory RAG index from given documents.
        """
        self._ensure_rag_components(embed_model=embed_model)
        if self.vector_store is None or self.embeddings_provider is None:
            raise ProviderRequestError("RAG components not initialized")

        all_chunks: list[str] = []
        metadatas: list[dict] = []
        for doc_idx, doc in enumerate(documents):
            chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
            for i, c in enumerate(chunks):
                all_chunks.append(c.text)
                metadatas.append({"doc_index": doc_idx, "chunk_index": i, "source": c.source})

        if not all_chunks:
            return

        embeddings = self.embeddings_provider.embed_batch(all_chunks)
        # ensure embeddings are a numpy array of correct dtype
        emb_array = self._to_np_array(embeddings, dtype="float32")

        chunks_for_store = []
        for text, meta in zip(all_chunks, metadatas, strict=False):
            chunk_id = f"doc_{meta['doc_index']}_chunk_{meta['chunk_index']}"
            if ids_prefix:
                chunk_id = f"{ids_prefix}{chunk_id}"
            chunks_for_store.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=meta["source"] if meta.get("source") is not None else "",
                    chunk_index=meta["chunk_index"],
                    metadata=meta,
                )
            )

        # Store embeddings and associated chunk objects in the vector store
        self.vector_store.add_embeddings(emb_array, chunks_for_store)

    def query_with_rag(
        self,
        query: str,
        k: int = 3,
        prompt_template: str | None = None,
        embed_model: str | None = None,
        temperature: float = 0.0,
    ) -> tuple[str, list[dict]]:
        """
        Perform a RAG query and return (answer, retrieved_contexts).
        """
        self._ensure_rag_components(embed_model=embed_model)
        if self.vector_store is None or self.embeddings_provider is None:
            raise ProviderRequestError("RAG components not initialized")

        q_emb = self.embeddings_provider.embed_batch([query])[0]
        q_arr = self._to_np_array(q_emb, dtype="float32")

        hits = self.vector_store.search(q_arr, top_k=k)

        contexts: list[dict] = []
        context_texts: list[str] = []
        for hit in hits:
            chunk = hit["chunk"]
            score = hit.get("score", 0.0)
            contexts.append({"score": score, "text": chunk.text, "metadata": chunk.metadata})
            context_texts.append(chunk.text)

        if not prompt_template:
            prompt_template = (
                "Use the following context to answer the question. If the context does not contain "
                "the answer, answer based on your knowledge and be explicit about missing info.\n\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )

        prompt = prompt_template.format(context="\n\n".join(context_texts), query=query)

        try:
            answer = self._create_chat_completion(prompt, temperature=temperature)
            return answer, contexts
        except Exception as exc:
            raise ProviderRequestError(f"Perplexity RAG request failed: {exc}") from exc
        
    def send(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                            )
        return response.choices[0].message.content
