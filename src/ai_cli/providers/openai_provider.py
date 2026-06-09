"""
OpenAI ChatGPT provider implementation for ai_cli with Advanced RAG support.

Optimized: batched embeddings, faster top-k retrieval, efficient append,
defensive checks, and safer save/load behavior.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any, Tuple

try:
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback not expected in normal use
    np = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - openai optional in some environments
    OpenAI = None  # type: ignore

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider
from ai_cli.providers.registry import register_provider

register_provider("openai", OpenAIProvider)

class OpenAIProvider(AIProvider):
    PROVIDER_NAME = "openai"
    PROVIDER_CLASS = OpenAIProvider
    BASE_URL = "https://api.openai.com/v1"
    DEFAULT_CHAT_MODEL = "gpt-5.5"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            provider_name="openai",
            model=model or self.DEFAULT_CHAT_MODEL,
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if OpenAI is None:
            raise ProviderRequestError(
                "The 'openai' package is not installed; install it with 'pip install openai'."
            )

        self.client = OpenAI(api_key=self.api_key)

        # In-memory vector store (numpy array of shape (N, D)) and metadata list
        self._vectors = None  # type: Optional[Any]
        self._metadatas: List[Dict[str, Any]] = []

    def _send_impl(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise ProviderRequestError("OpenAI returned no completion choices")
            message = choices[0].message
            content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
            if not content:
                raise ProviderRequestError("OpenAI returned empty response content")
            return content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": "ping"}], max_tokens=5
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
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            overlap = 0
        words = text.split()
        chunks: List[str] = []
        start = 0
        n = len(words)
        while start < n:
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            if end >= n:
                break
            start = end - overlap
        return chunks

    def _create_embeddings(self, inputs: List[str], model: Optional[str] = None, batch_size: int = 256) -> List[List[float]]:
        """
        Create embeddings for inputs in batches. Returns list of embeddings.
        """
        if np is None:
            raise ProviderRequestError("NumPy is required for embedding operations. Install numpy.")
        model = model or self.DEFAULT_EMBEDDING_MODEL
        embeddings: List[List[float]] = []
        # Batch to avoid request size limits
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            try:
                resp = self.client.embeddings.create(model=model, input=batch)
            except Exception as exc:
                raise ProviderRequestError(f"Embedding request failed: {exc}") from exc
            for item in resp.data:
                emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
                if emb is None:
                    raise ProviderRequestError("Embedding response missing embedding field")
                embeddings.append(list(emb))
        return embeddings

    def embed_chunks(self, chunks: List[str], embedding_model: Optional[str] = None) -> List[List[float]]:
        return self._create_embeddings(chunks, model=embedding_model)

    def build_vector_store(
        self,
        documents: List[Dict[str, str]],
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_model: Optional[str] = None,
    ) -> None:
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

        embeddings = self._create_embeddings(all_chunks, model=embedding_model)

        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")

        vecs = np.asarray(embeddings, dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        if self._vectors is None:
            self._vectors = vecs
            self._metadatas = metadatas
        else:
            # efficient concatenate
            self._vectors = np.concatenate((self._vectors, vecs), axis=0)
            self._metadatas.extend(metadatas)

    def query_vector_store(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")
        if self._vectors is None or len(self._metadatas) == 0:
            return []

        q = np.asarray(query_embedding, dtype=float)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0
        q = q / q_norm

        scores = self._vectors @ q  # shape (N,)
        n = scores.shape[0]
        k = min(max(1, top_k), n)

        # Fast top-k: argpartition then sort those top elements
        if k < n:
            idxs = np.argpartition(-scores, k - 1)[:k]
            top_idxs = idxs[np.argsort(-scores[idxs])]
        else:
            top_idxs = np.argsort(-scores)

        results: List[Tuple[float, Dict[str, Any]]] = []
        for idx in top_idxs[:k]:
            results.append((float(scores[idx]), self._metadatas[int(idx)]))
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
        if self._vectors is None or len(self._metadatas) == 0:
            raise ProviderRequestError("Vector store is empty. Call build_vector_store first.")

        try:
            emb_resp = self.client.embeddings.create(
                model=embedding_model or self.DEFAULT_EMBEDDING_MODEL, input=[query]
            )
            item = emb_resp.data[0]
            query_emb = getattr(item, "embedding", None) or (item.get("embedding") if isinstance(item, dict) else None)
            if query_emb is None:
                raise ProviderRequestError("Embedding response missing embedding field")
        except Exception as exc:
            raise ProviderRequestError(f"Embedding request failed: {exc}") from exc

        hits = self.query_vector_store(query_emb, top_k=top_k)
        if not hits:
            return self._send_impl(query)

        contexts = [f"[{meta.get('source_id')}#chunk{meta.get('chunk_index')}] {meta.get('text')}" for _, meta in hits]
        context_text = "\n\n---\n\n".join(contexts)

        if prompt_template is None:
            prompt = (
                "You are an assistant that answers user queries using the provided context. "
                "If the answer is not contained in the context, say you don't know.\n\n"
                "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer concisely:"
            ).format(context=context_text, query=query)
        else:
            prompt = prompt_template.format(context=context_text, query=query)

        return self._send_impl(prompt)

    def save_vector_store(self, path: str) -> None:
        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")
        if self._vectors is None:
            raise ProviderRequestError("No vector store to save")
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # store as float32 to reduce size
            np.savez_compressed(path, vectors=self._vectors.astype(np.float32), metadatas=np.array(self._metadatas, dtype=object))
        except Exception as exc:
            raise ProviderRequestError(f"Failed to save vector store: {exc}") from exc

    def load_vector_store(self, path: str) -> None:
        if np is None:
            raise ProviderRequestError("NumPy is required for vector store operations. Install numpy.")
        try:
            data = np.load(path, allow_pickle=True)
            vectors = data["vectors"].astype(float)
            metadatas = data["metadatas"].tolist()
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
            self._vectors = vectors
            self._metadatas = metadatas
        except Exception as exc:
            raise ProviderRequestError(f"Failed to load vector store: {exc}") from exc

class OpenAIEmbeddingProvider:
    def __init__(self, model="text-embedding-3-small", api_key=None):
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts):
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [d.embedding for d in resp.data]
