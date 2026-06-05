"""
xAI Grok provider implementation for ai_cli with Advanced RAG support.

This module integrates xAI Grok models into the ai_cli provider framework using
xAI's OpenAI-compatible API and adds utilities for Retrieval-Augmented Generation (RAG):
- text chunking with overlap
- embedding creation via xAI embeddings endpoint
- an in-memory vector DB (with cosine similarity) for storage and retrieval
- helper routines to upsert documents and perform retrieval + generation.

Environment Variables
---------------------
XAI_API_KEY
    API key used to authenticate with xAI API.

Example
-------
export XAI_API_KEY="your_api_key"

Usage
-----
provider = XAIProvider(
    model="grok-2-latest",
    embedding_model="text-embedding-3-small"
)

# Add long documents (they will be chunked, embedded and stored)
provider.add_documents(["Long document text ...", "Another doc ..."])

# Regular prompt (no RAG)
response = provider.send("Explain Kubernetes operators")

# RAG-enabled generation (retrieval + generation)
response_with_context = provider.send_rag("How do operators handle CRDs?", top_k=4)
print(response_with_context)
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

# Import the shared base provider and error type from the package.
# Provide robust fallbacks so the module can still be imported in isolation
# (useful for linters/tests) while preferring the project's definitions.
try:
    from .base import AIProvider, ProviderRequestError
except Exception:
    try:
        from ai_cli.providers.base import AIProvider, ProviderRequestError
    except Exception:
        class ProviderRequestError(Exception):
            pass

        class AIProvider:
            def __init__(self, *args, **kwargs):
                # reference args to avoid unused-variable lint warnings in fallback
                _ = args
                self.model = kwargs.get("model") if kwargs else None

# numpy import: keep static typing happy and provide a runtime-safe fallback
if TYPE_CHECKING:
    import numpy as np  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

# OpenAI / xAI client import with a safe fallback for environments without the package.
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


class InMemoryVectorStore:
    """
    Simple in-memory vector store using numpy arrays.

    - vectors: np.ndarray shape (n, d)
    - metadatas: list of dicts (contains 'text' and optional 'meta')
    """

    def __init__(self) -> None:
        # lazy import guard: numpy may be missing in some environments
        # use Any for runtime-safe typing to avoid referencing the `np` variable
        self._vectors: Optional[Any] = None
        self._metadatas: List[Dict[str, Any]] = []

    def upsert(self, embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        if np is None:
            raise RuntimeError("numpy is required for InMemoryVectorStore. Install with `pip install numpy`")
        if len(embeddings) != len(metadatas):
            raise ValueError("embeddings and metadatas must have the same length")
        arr = np.array(embeddings, dtype=np.float32)
        if self._vectors is None:
            self._vectors = arr
            self._metadatas = list(metadatas)
        else:
            # stack new embeddings onto existing matrix and extend metadatas
            self._vectors = np.vstack([self._vectors, arr])
            self._metadatas.extend(metadatas)
    def _cosine_similarity(self, q: Any, vecs: Any) -> Any:
        if np is None:
            raise RuntimeError("numpy is required for InMemoryVectorStore. Install with `pip install numpy`")
        # q: (d,), vecs: (n, d)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
        sims = vecs_norm.dot(q_norm)
        return sims
        return sims

    def query(self, embedding: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if np is None:
            raise RuntimeError("numpy is required for InMemoryVectorStore. Install with `pip install numpy`")
        if self._vectors is None or len(self._metadatas) == 0:
            return []
        q = np.array(embedding, dtype=np.float32)
        sims = self._cosine_similarity(q, self._vectors)
        idx = np.argsort(-sims)[:top_k]
        results: List[Tuple[Dict[str, Any], float]] = []
        for i in idx:
            results.append((self._metadatas[int(i)], float(sims[int(i)])))
        return results

    def count(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])


class XAIProvider(AIProvider):
    """
    AI provider implementation for xAI Grok models with added RAG helpers.

    This provider communicates with xAI's OpenAI-compatible chat completions and
    embeddings APIs.

    New RAG-related parameters
    --------------------------
    embedding_model : str
        Model used to create embeddings (defaults to "text-embedding-3-small").

    vector_store : Optional[InMemoryVectorStore]
        Custom vector DB instance. If None, an in-memory vector store is used.

    Features
    --------
    - chunk_text: split long documents into chunks with overlap
    - add_documents: chunk, embed and upsert documents into the vector store
    - send_rag: retrieve top-k relevant chunks and generate an answer using context
    """

    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            provider_name="xai",
            model=model or "grok-2-latest",
            api_key=api_key,
            *args,
            **kwargs,
        )
        # OpenAI-compatible client
        if OpenAI is None:
            raise ProviderRequestError("openai package is required. Install with `pip install openai`")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )
        # RAG configuration
        self.embedding_model = embedding_model or "text-embedding-3-small"
        self.vector_store = vector_store or InMemoryVectorStore()

    # --------------------
    # Chunking utilities
    # --------------------
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Chunk text into pieces of approximately chunk_size characters with overlap.

        This is a simple character-based chunker that tries to avoid breaking in the
        middle of words when possible.

        Parameters
        ----------
        text : str
        chunk_size : int
        overlap : int

        Returns
        -------
        List[str]
            List of text chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        chunks: List[str] = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Try to avoid splitting in the middle of a word if not at end
            if end < text_len and text[end] not in (" ", "\n"):
                # backtrack to last space within a small window
                back = text.rfind(" ", start, end)
                if back > start:
                    end = back
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = max(start + 1, end - overlap) if end < text_len else end
        return chunks

    # --------------------
    # Embedding utilities
    # --------------------
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using xAI embeddings endpoint.

        Returns list of embedding vectors (list of floats).
        """
        try:
            # xAI/OpenAI-compatible embeddings create
            resp = self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
            )
            # resp.data is a list matching texts
            embeddings: List[List[float]] = [item.embedding for item in resp.data]
            return embeddings
        except Exception as exc:
            raise ProviderRequestError(f"xAI embeddings request failed: {exc}") from exc

    # --------------------
    # Vector store helpers
    # --------------------
    def add_documents(self, docs: List[str], chunk_size: int = 1000, overlap: int = 200) -> int:
        """
        Chunk documents, embed chunks and upsert into the vector store.

        Returns number of chunks inserted.
        """
        all_chunks: List[str] = []
        for doc in docs:
            chunks = self.chunk_text(doc, chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
        if not all_chunks:
            return 0
        embeddings = self.embed_texts(all_chunks)
        metadatas = [{"text": txt} for txt in all_chunks]
        self.vector_store.upsert(embeddings, metadatas)
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k most relevant chunks for the query.

        Returns list of metadata dicts (contains 'text') in order of relevance.
        """
        emb = self.embed_texts([query])
        if not emb:
            return []
        results = self.vector_store.query(emb[0], top_k=top_k)
        return [r[0] for r in results]

    # --------------------
    # Generation (standard + RAG)
    # --------------------
    def _call_model(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Internal helper: send prompt (with optional system message) to the Grok model.
        """
        try:
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0.7)

            if not getattr(response, "choices", None):
                raise ProviderRequestError("xAI returned no completion choices")
            message = response.choices[0].message

            if not message or not getattr(message, "content", None):
                raise ProviderRequestError("xAI returned empty response content")
            return message.content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"xAI request failed: {exc}") from exc

    def _send_impl(self, prompt: str) -> str:
        """Send prompt to Grok model (required by AIProvider base class)."""
        return self._call_model(prompt)

    def send_rag(self, prompt: str, top_k: int = 4, instruction: Optional[str] = None) -> str:
        """
        Perform RAG: retrieve relevant chunks and generate an answer grounded on them.

        Parameters
        ----------
        prompt : str
            User question to answer.
        top_k : int
            Number of retrieved chunks to include.
        instruction : Optional[str]
            Optional instructions for the model about how to use the retrieved context.

        Returns
        -------
        str
            Generated response text.
        """
        try:
            retrieved = self.retrieve(prompt, top_k=top_k)
            if not retrieved:
                # Fallback to normal generation if no context available
                return self._send_impl(prompt)

            context_parts = [f"Source {i+1}:\n{item.get('text', '')}" for i, item in enumerate(retrieved)]
            context = "\n\n".join(context_parts)

            # Build a clear system prompt if provided, else a default one that instructs the model to use context.
            system_prompt = instruction or (
                "You are a helpful assistant. Use the provided context to answer the user's question. "
                "Cite sources when appropriate. If the context is insufficient, say so and do not hallucinate."
            )

            # Craft combined prompt with context and user question
            augmented_prompt = f"Context:\n{context}\n\nUser question:\n{prompt}\n\nAnswer using the context above."

            return self._call_model(augmented_prompt, system_prompt=system_prompt)
        except Exception as exc:
            raise ProviderRequestError(f"RAG request failed: {exc}") from exc

    def health_check(self) -> bool:
        """
        Perform lightweight connectivity test.

        Returns True if provider is operational, otherwise False.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "ping"},
                ],
                max_tokens=5,
            )
            return bool(getattr(response, "choices", None))
        except Exception:
            return False