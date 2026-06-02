from __future__ import annotations
import os
import logging
from typing import List, Tuple, Optional, Any, Dict

# Optional imports for embeddings/vector math
try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

from ai_cli.providers.base import AIProvider
from ai_cli.providers.registry import PROVIDERS, register_provider
from ai_cli.core.exceptions import (
    ProviderConfigurationError,
    ProviderRequestError,
    ResponseValidationError,
)

logger = logging.getLogger(__name__)

"""
Advanced RAG utilities added here:

- Chunking: chunk_text(text, chunk_size=500, overlap=50)
- Embedding: provider.get_embeddings(texts) implemented for OpenAI and OpenAI-compatible providers
- Vector store: SimpleVectorStore (in-memory, numpy-backed brute-force search)
- RAGManager: helpers to build vector stores from documents and query them

Notes:
- numpy is optional; if not installed, embedding/vector operations that require numpy will raise a clear error.
- Embedding model names are defaults; you can override by specifying provider.model when constructing providers.
- This file also fixes previous indentation/method placement issues for OpenAICompatibleProvider.
"""


class OpenAIProvider(AIProvider):
    """Concrete provider using OpenAI SDK."""

    def __init__(self, model: str | None = None) -> None:
        super().__init__(
            provider_name="openai",
            model=model,
            provider_meta=PROVIDERS["openai"],
        )

    def _send_impl(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError("Install openai package") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key, timeout=self.timeout)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

        usage = getattr(response, "usage", None)
        if usage:
            self.metrics.total_prompt_tokens += getattr(
                usage, "prompt_tokens", 0
            )
            self.metrics.total_completion_tokens += getattr(
                usage, "completion_tokens", 0
            )

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ResponseValidationError("Invalid response structure") from exc
        if not content or not isinstance(content, str):
            raise ResponseValidationError("Empty response")
        return content.strip()

    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Return embeddings for a list of texts using OpenAI embeddings API.
        """
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError("Install openai package") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key, timeout=self.timeout)
        emb_model = model or (self.model if self.model else "text-embedding-3-small")
        try:
            # The OpenAI SDK may vary in API shape; handle common shape
            response = client.embeddings.create(model=emb_model, input=texts)
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI embeddings request failed: {exc}") from exc

        # Response shape: data[i].embedding
        try:
            embeddings = [d.embedding for d in response.data]
        except Exception as exc:
            raise ResponseValidationError("Invalid embeddings response structure") from exc
        return embeddings


class OpenAICompatibleProvider(AIProvider):
    """Generic provider for OpenAI-compatible APIs."""

    api_base_url: str = ""
    api_key_env: str = ""

    def __init__(self, provider_name: str, model: str | None = None) -> None:
        super().__init__(
            provider_name=provider_name,
            model=model,
            provider_meta=PROVIDERS[provider_name],
        )

    def _send_impl(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install OpenAI SDK: pip install openai"
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderConfigurationError(f"{self.api_key_env} not set")

        client = OpenAI(api_key=api_key, base_url=self.api_base_url, timeout=self.timeout)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"{self.provider_name} request failed: {exc}"
            ) from exc

        usage = getattr(response, "usage", None)
        if usage:
            self.metrics.total_prompt_tokens += getattr(
                usage, "prompt_tokens", 0
            )
            self.metrics.total_completion_tokens += getattr(
                usage, "completion_tokens", 0
            )

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ResponseValidationError("Invalid response structure") from exc
        if not content or not isinstance(content, str):
            raise ResponseValidationError("Empty response")
        return content.strip()

    def get_embeddings(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """
        Return embeddings for a list of texts using the OpenAI-compatible embeddings endpoint.
        """
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError("Install OpenAI SDK: pip install openai") from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderConfigurationError(f"{self.api_key_env} not set")

        client = OpenAI(api_key=api_key, base_url=self.api_base_url, timeout=self.timeout)
        emb_model = model or (self.model if self.model else "text-embedding-3-small")
        try:
            response = client.embeddings.create(model=emb_model, input=texts)
        except Exception as exc:
            raise ProviderRequestError(f"{self.provider_name} embeddings request failed: {exc}") from exc

        try:
            embeddings = [d.embedding for d in response.data]
        except Exception as exc:
            raise ResponseValidationError("Invalid embeddings response structure") from exc
        return embeddings


class PerplexityProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.perplexity.ai"
    api_key_env = "PERPLEXITY_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="perplexity", model=model)


class DeepSeekProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="deepseek", model=model)


class GroqProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="groq", model=model)


class OpenRouterProvider(OpenAICompatibleProvider):
    api_base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="openrouter", model=model)


class TogetherProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.together.xyz/v1"
    api_key_env = "TOGETHER_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="together", model=model)


class FireworksProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.fireworks.ai/inference/v1"
    api_key_env = "FIREWORKS_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="fireworks", model=model)


class XAIProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.x.ai/v1"
    api_key_env = "XAI_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="xai", model=model)


class GeminiProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.gemini.google/v1"
    api_key_env = "GEMINI_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="gemini", model=model)


class CohereProvider(AIProvider):
    api_key_env = "COHERE_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        super().__init__(
            provider_name="cohere",
            model=model,
            provider_meta=PROVIDERS["cohere"],
        )

    def _send_impl(self, prompt: str) -> str:
        try:
            import cohere
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install cohere package: pip install cohere"
            ) from exc
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderConfigurationError(f"{self.api_key_env} not set")

        client = cohere.Client(api_key)
        try:
            response = client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(f"Cohere request failed: {exc}") from exc

        try:
            content = response.generations[0].text
        except Exception as exc:
            raise ResponseValidationError("Invalid response structure") from exc

        if not content or not isinstance(content, str):
            raise ResponseValidationError("Empty response")

        # Try to update token metrics if available (best-effort)
        try:
            token_count = getattr(response, "token_count", None) or (
                (getattr(response, "meta", {}) or {}).get("token_count")
            )
            if token_count is not None:
                self.metrics.total_prompt_tokens += int(token_count)
        except Exception as exc:
            logger.debug("Token metric update failed: %s", exc, exc_info=True)

        return content.strip()


# ---- RAG helpers ----
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks of approx chunk_size characters.
    Overlap ensures context continuity.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap
    return chunks


class SimpleVectorStore:
    """
    Minimal in-memory vector store using numpy.
    - Add vectors and metadata
    - Query by cosine similarity (brute-force)

    Requires numpy. Designed for small-to-moderate datasets and demonstration/testing.
    """

    def __init__(self) -> None:
        if np is None:
            raise ProviderConfigurationError("numpy is required for SimpleVectorStore")
        self._vectors = np.zeros((0, 0), dtype=float)  # shape (n, dim)
        self._metadatas: List[Dict[str, Any]] = []
        self._dim: Optional[int] = None

    def add(self, vectors: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if not vectors:
            return
        arr = np.array(vectors, dtype=float)
        if self._dim is None:
            self._dim = arr.shape[1]
            self._vectors = arr
        else:
            if arr.shape[1] != self._dim:
                raise ValueError("All vectors must have same dimensionality")
            self._vectors = np.vstack([self._vectors, arr])
        if metadatas:
            self._metadatas.extend(metadatas)
        else:
            self._metadatas.extend([{} for _ in range(len(vectors))])

    @staticmethod
    def _cosine_similarity_matrix(a: Any, b: Any) -> Any:
        # a: (m, d), b: (n, d) -> returns (m, n) as a numpy.ndarray
        # Use Any for type hints to avoid referencing the numpy variable in type expressions
        # when numpy may not be present at type checking / import time.
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        # avoid division by zero
        a_norm[a_norm == 0] = 1.0
        b_norm[b_norm == 0] = 1.0
        sim = (a @ b.T) / (a_norm * b_norm.T)
        return sim

    def query(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self._dim is None or self._vectors.shape[0] == 0:
            return []
        q = np.array([query_vector], dtype=float)
        if q.shape[1] != self._dim:
            raise ValueError("Query vector dimensionality does not match store")
        sims = self._cosine_similarity_matrix(q, self._vectors)[0]  # shape (n,)
        # get top_k indices
        top_k = max(1, min(top_k, len(sims)))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        idx_sorted = idx[np.argsort(-sims[idx])]
        results: List[Tuple[Dict[str, Any], float]] = []
        for i in idx_sorted:
            results.append((self._metadatas[i], float(sims[i])))
        return results


class RAGManager:
    """
    High-level helper to build a vector store from documents and perform retrieval.
    Usage:
      - Instantiate with a provider that implements get_embeddings(texts)
      - build_store_from_texts(documents, chunk_size=500, overlap=50)
      - query(prompt, top_k=5) -> returns list of (metadata, score, chunk_text)
    """

    def __init__(self, embedding_provider: AIProvider):
        if not hasattr(embedding_provider, "get_embeddings"):
            raise ValueError("Provider must implement get_embeddings(texts: List[str])")
        if np is None:
            raise ProviderConfigurationError("numpy is required for RAGManager")
        self.provider = embedding_provider
        self.store = SimpleVectorStore()

    def build_store_from_texts(
        self,
        texts: List[str],
        chunk_size: int = 500,
        overlap: int = 50,
        metadata_fn: Optional[Any] = None,
    ) -> None:
        """
        Chunk each text and embed the chunks; metadata_fn, if provided, should accept (text_index, chunk_index, chunk_str)
        and return a metadata dict.
        """
        all_chunks: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        for ti, text in enumerate(texts):
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for ci, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = metadata_fn(ti, ci, chunk) if metadata_fn else {"doc_index": ti, "chunk_index": ci}
                metadatas.append(meta)
        if not all_chunks:
            return
        embeddings = self.provider.get_embeddings(all_chunks)
        self.store.add(embeddings, metadatas)

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float, str]]:
        """
        Return list of (metadata, score, chunk_text) for top_k nearest chunks.
        """
        embeddings = self.provider.get_embeddings([query_text])
        if not embeddings:
            return []
        q_vec = embeddings[0]
        results = self.store.query(q_vec, top_k=top_k)
        # enrich results with chunk text if available
        enriched: List[Tuple[Dict[str, Any], float, str]] = []
        # We don't store raw chunks in SimpleVectorStore; reconstruct via metadatas if they included chunk text
        for meta, score in results:
            chunk_text_val = meta.get("chunk_text") or meta.get("text") or ""
            enriched.append((meta, score, chunk_text_val))
        return enriched


# Register them
register_provider("openai", OpenAIProvider, PROVIDERS["openai"])
register_provider("perplexity", PerplexityProvider, PROVIDERS["perplexity"])
register_provider("cohere", CohereProvider, PROVIDERS["cohere"])
register_provider("deepseek", DeepSeekProvider, PROVIDERS["deepseek"])
register_provider("groq", GroqProvider, PROVIDERS["groq"])
register_provider("openrouter", OpenRouterProvider, PROVIDERS["openrouter"])
register_provider("together", TogetherProvider, PROVIDERS["together"])
register_provider("fireworks", FireworksProvider, PROVIDERS["fireworks"])
register_provider("xai", XAIProvider, PROVIDERS["xai"])
register_provider("gemini", GeminiProvider, PROVIDERS["gemini"])