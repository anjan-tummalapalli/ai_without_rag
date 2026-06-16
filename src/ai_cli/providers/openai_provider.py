"""
OpenAI ChatGPT provider implementation for ai_cli with RAG support.
"""
from __future__ import annotations

import os
from typing import Any

try:
    import numpy as np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import BaseProvider
from ai_cli.providers.registry import register_chat_provider, register_provider


class OpenAIProvider(BaseProvider):
    PROVIDER_NAME = "openai"
    DEFAULT_CHAT_MODEL = "gpt-5.5"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")
        if OpenAI is None:
            raise ProviderRequestError(
                "The 'openai' package is not installed. Install it via "
                "'pip install openai'."
            )
        self.client = OpenAI(api_key=self.api_key)
        self._vectors: Any | None = None
        self._metadatas: list[dict[str, Any]] = []

    def _send_impl(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise ProviderRequestError("OpenAI returned no choices")
            first = choices[0]
            message = getattr(first, "message", None) or (
                first.get("message") if isinstance(first, dict) else None
            )
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if not content:
                raise ProviderRequestError("OpenAI returned empty content")
            return content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(getattr(resp, "choices", None))
        except Exception:
            return False

    def send(self, prompt: str, **kwargs: Any) -> str:
        return self._send_impl(prompt)

    # ----------------------------
    # RAG / embedding helpers
    # ----------------------------
    def chunk_text(
        self, text: str, chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            overlap = 0
        words = text.split()
        chunks: list[str] = []
        start = 0
        n = len(words)
        while start < n:
            end = start + chunk_size
            chunks.append(" ".join(words[start:end]))
            if end >= n:
                break
            start = end - overlap
        return chunks

    def _create_embeddings(
        self,
        inputs: list[str],
        model: str | None = None,
        batch_size: int = 256,
    ) -> list[list[float]]:
        if np is None:
            raise ProviderRequestError(
                "NumPy is required for embeddings. Install numpy."
            )
        model = model or self.DEFAULT_EMBEDDING_MODEL
        embeddings: list[list[float]] = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            try:
                resp = self.client.embeddings.create(model=model, input=batch)
            except Exception as exc:
                raise ProviderRequestError(
                    f"Embedding request failed: {exc}"
                ) from exc
            data = getattr(resp, "data", None) or []
            for item in data:
                emb = getattr(item, "embedding", None) or (
                    item.get("embedding") if isinstance(item, dict) else None
                )
                if emb is None:
                    raise ProviderRequestError(
                        "Embedding response missing field"
                    )
                embeddings.append(list(emb))
        return embeddings

    def build_vector_store(
        self,
        documents: list[dict[str, str]],
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_model: str | None = None,
    ) -> None:
        all_chunks: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for doc in documents:
            doc_id = doc.get("id") or doc.get("source") or "unknown"
            text = doc.get("text", "")
            chunks = self.chunk_text(
                text, chunk_size=chunk_size, overlap=overlap
            )
            for idx, chunk in enumerate(chunks):
                metadatas.append(
                    {"source_id": doc_id, "chunk_index": idx, "text": chunk}
                )
                all_chunks.append(chunk)

        if not all_chunks:
            return

        embeddings = self._create_embeddings(
            all_chunks, model=embedding_model
        )

        if np is None:
            raise ProviderRequestError(
                "NumPy is required for vector store operations."
            )

        vecs = np.asarray(embeddings, dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms

        if self._vectors is None:
            self._vectors = vecs
            self._metadatas = metadatas
        else:
            self._vectors = np.concatenate((self._vectors, vecs), axis=0)
            self._metadatas.extend(metadatas)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        embedding_model: str | None = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        if np is None:
            raise ProviderRequestError(
                "NumPy is required for retrieval operations."
            )
        if self._vectors is None or len(self._metadatas) == 0:
            raise ProviderRequestError(
                "Vector store is empty. Call build_vector_store first."
            )

        try:
            emb_resp = self.client.embeddings.create(
                model=embedding_model or self.DEFAULT_EMBEDDING_MODEL,
                input=[query],
            )
            item = getattr(emb_resp, "data", [])[0]
            query_emb = getattr(item, "embedding", None) or (
                item.get("embedding") if isinstance(item, dict) else None
            )
            if query_emb is None:
                raise ProviderRequestError(
                    "Embedding response missing field"
                )
        except Exception as exc:
            raise ProviderRequestError(
                f"Embedding request failed: {exc}"
            ) from exc

        q = np.asarray(query_emb, dtype=float)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            q_norm = 1.0
        q = q / q_norm

        scores = self._vectors @ q
        n = scores.shape[0]
        k = min(max(1, top_k), n)

        if k < n:
            idxs = np.argpartition(-scores, k - 1)[:k]
            top_idxs = idxs[np.argsort(-scores[idxs])]
        else:
            top_idxs = np.argsort(-scores)

        results: list[tuple[float, dict[str, Any]]] = []
        for idx in top_idxs[:k]:
            results.append((float(scores[idx]), self._metadatas[int(idx)]))
        return results

    def save_vector_store(self, path: str) -> None:
        if np is None:
            raise ProviderRequestError(
                "NumPy is required for vector store operations."
            )
        if self._vectors is None:
            raise ProviderRequestError("No vector store to save")
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.savez_compressed(
                path,
                vectors=self._vectors.astype(np.float32),
                metadatas=np.array(self._metadatas, dtype=object),
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"Failed to save vector store: {exc}"
            ) from exc

    def load_vector_store(self, path: str) -> None:
        if np is None:
            raise ProviderRequestError(
                "NumPy is required for vector store operations."
            )
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
            raise ProviderRequestError(
                f"Failed to load vector store: {exc}"
            ) from exc

    def ask(self, prompt: str, **kwargs: Any) -> str:
        return self.send(prompt, **kwargs)


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        if OpenAI is None:
            raise ProviderRequestError(
                "The 'openai' package is not installed. Install it via "
                "'pip install openai'."
            )
        self.client = OpenAI(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self.client.embeddings.create(
                model=self.model, input=texts
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"Embedding request failed: {exc}"
            ) from exc
        out: list[list[float]] = []
        for d in getattr(resp, "data", []):
            emb = getattr(d, "embedding", None) or (
                d.get("embedding") if isinstance(d, dict) else None
            )
            if emb is None:
                raise ProviderRequestError(
                    "Embedding response missing field"
                )
            out.append(list(emb))
        return out


register_provider("openai", OpenAIProvider)
register_chat_provider("openai", OpenAIProvider)