"""
Perplexity provider implementation with optional Advanced RAG support.

This module integrates Perplexity AI models into the ai_cli provider framework
using Perplexity's OpenAI-compatible API and adds utilities to:

- chunk documents
- embed chunks
- store embeddings in a lightweight in-memory vector DB
- retrieve context and perform RAG-style calls

Environment Variables
---------------------
PERPLEXITY_API_KEY
    API key used to authenticate with Perplexity API.

Usage
-----
provider = PerplexityProvider(model="sonar-pro")
provider.build_rag_index(["Long document text..."])
answer = provider.query_with_rag("What is the summary?", k=3)
print(answer)
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None
from openai import OpenAI

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider
from ai_cli.rag.chunker import chunk_text
from ai_cli.rag.embeddings import EmbeddingsProvider
from ai_cli.rag.vector_store import InMemoryVectorStore


class PerplexityProvider(AIProvider):
    BASE_URL = "https://api.perplexity.ai"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(model=model or "sonar-pro", api_key=api_key, *args, **kwargs)
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ProviderRequestError("PERPLEXITY_API_KEY environment variable is not set")

        # OpenAI-compatible client
            self.embeddings_provider: Optional[EmbeddingsProvider] = None
            self.vector_store: Optional[InMemoryVectorStore] = None
    
        def _to_np_array(self, data, dtype=None):
            """
            Helper to convert data to a numpy array, or raise a clear error if numpy
            is not available so callers get an actionable message instead of a
            mysterious import/linter error.
            """
            if np is None:
                raise ProviderRequestError(
                    "numpy is required for RAG features; please install it (pip install numpy)"
                )
            return np.array(data, dtype=dtype)
        # RAG components (lazy-init)
        self.embeddings_provider: Optional[EmbeddingsProvider] = None
        self.vector_store: Optional[InMemoryVectorStore] = None

    def _send_impl(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            if not getattr(response, "choices", None):
                raise ProviderRequestError("Perplexity returned no choices")
            message = response.choices[0].message
            if not message or not getattr(message, "content", None):
                raise ProviderRequestError("Perplexity returned empty content")
            return message.content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"Perplexity request failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": "ping"}]
            )
            return bool(getattr(response, "choices", None))
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "perplexity"

    # -------------------------
    # Advanced RAG helpers
    # -------------------------

    def _ensure_rag_components(self, embed_model: Optional[str] = None) -> None:
        if self.embeddings_provider is None:
            self.embeddings_provider = EmbeddingsProvider(self.client, model=embed_model)
        if self.vector_store is None:
            self.vector_store = InMemoryVectorStore()

    def build_rag_index(
        self,
        documents: Sequence[str],
        chunk_size: int = 500,
        overlap: int = 50,
        embed_model: Optional[str] = None,
        ids_prefix: Optional[str] = None,
    ) -> None:
        """
        Build an in-memory RAG index from given documents.

        - chunks documents with overlap
        - computes embeddings for chunks
        - stores embeddings and chunk text/metadata in the vector store
        """
        embeddings = self.embeddings_provider.embed_texts(all_chunks)
        self.vector_store.add(self._to_np_array(embeddings, dtype=getattr(np, "float32", None)), all_chunks, metadatas, ids)
        all_chunks = []
        metadatas = []
        ids = []
        for doc_idx, doc in enumerate(documents):
            chunks = chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
            for i, c in enumerate(chunks):
                all_chunks.append(c)
                metadatas.append({"doc_index": doc_idx, "chunk_index": i})
                ids.append(f"{ids_prefix or 'doc'}-{doc_idx}-{i}")

        if not all_chunks:
            return

        embeddings = self.embeddings_provider.embed_texts(all_chunks)
        self.vector_store.add(np.array(embeddings, dtype=np.float32), all_chunks, metadatas, ids)

    def query_with_rag(
        self,
        query: str,
        k: int = 3,
        prompt_template: Optional[str] = None,
        embed_model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> Tuple[str, List[dict]]:
        """
        Perform a RAG query:
        - embed the user query
        - retrieve top-k relevant chunks
        - construct a prompt combining the retrieved context and the query
        - call the model and return (answer, retrieved_contexts)

        Returns:
            answer (str), retrieved_contexts (list of metadata+text)
        """
        self._ensure_rag_components(embed_model=embed_model)
        if self.vector_store is None or self.embeddings_provider is None:
            raise ProviderRequestError("RAG components not initialized")

        q_emb = self.embeddings_provider.embed_texts([query])[0]
        hits = self.vector_store.search(np.array(q_emb, dtype=np.float32), k=k)

        contexts = []
        context_text = []
        for score, text, metadata in hits:
            contexts.append({"score": float(score), "text": text, "metadata": metadata})
            context_text.append(text)

        if prompt_template is None:
            prompt_template = (
                "Use the following context to answer the question. If the context does not contain "
                "the answer, answer based on your knowledge and be explicit about missing info.\n\n"
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            )
        prompt = prompt_template.format(context="\n\n".join(context_text), query=query)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if not getattr(response, "choices", None):
                raise ProviderRequestError("Perplexity returned no choices")
            answer = response.choices[0].message.content.strip()
            return answer, contexts
        except Exception as exc:
            raise ProviderRequestError(f"Perplexity RAG request failed: {exc}") from exc
