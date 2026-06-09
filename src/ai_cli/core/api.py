from __future__ import annotations
import os
import logging
from typing import List, Sequence, Dict, Any, Optional, Tuple

from ai_cli.providers.registry import load_plugins
from ai_cli.providers.registry import build_provider, PROVIDERS
from ai_cli.providers.registry import get_chat_provider
from ai_cli.core.exceptions import AIProviderError
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider
from ai_cli.providers.registry import get_chat_provider
from ai_cli.providers.bootstrap import init_providers

CHAT_PROVIDERS = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "cohere": CohereProvider,
    "deepseek": DeepSeekProvider,
    "perplexity": PerplexityProvider,
    "xai": XAIProvider,
    "zai": ZAIProvider,
}

def get_chat_provider(name: str, **kwargs):
    key = name.lower().strip()
    if key not in CHAT_PROVIDERS:
        raise ValueError(f"Unknown chat provider: {name}")
    
    return CHAT_PROVIDERS[key](**kwargs)

logger = logging.getLogger("ai_gateway")

# Ensure plugins are loaded at import time
load_plugins()


def ask(
    prompt: str,
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    embedding_model: str | None = None,
    timeout: float | None = None,
):
    init_providers()
    import os

    if api_key is None:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")

    # FIX HERE (AUTO RESOLUTION)
    if provider == "auto":
        provider = "openai"

    ai_provider = build_provider(
                                 name=provider,
                                 model=model,
                                 api_key=api_key,
                                )


# --- Advanced RAG helpers: chunking, embeddings, in-memory vector DB querying ---


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Simple text chunking that splits by whitespace to produce overlapping chunks.

    - chunk_size: approximate max characters per chunk
    - chunk_overlap: number of characters to overlap between adjacent chunks
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= length:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _compute_embeddings(
    ai_provider,
    texts: Sequence[str],
) -> List[List[float]]:
    """
    Compute embeddings via provider.

    The provider is expected to implement one of:
    - embed(text: str) -> List[float]
    - embed_many(texts: Sequence[str]) -> List[List[float]]
    or
    - embed_batch(texts: Sequence[str]) -> List[List[float]]

    Raises AIProviderError if provider does not support embedding.
    """
    if not texts:
        return []

    # Prefer batch APIs
    for method_name in ("embed_many", "embed_batch", "embed"):
        method = getattr(ai_provider, method_name, None)
        if method:
            break
    else:
        raise AIProviderError("provider does not support embeddings")

    try:
        if method_name == "embed":
            # call per-text
            return [method(t) for t in texts]
        else:
            return method(texts)
    except Exception as exc:
        logger.exception("embedding_failure error=%s", exc)
        raise AIProviderError(f"embedding failed: {exc}") from exc


def _build_in_memory_index(
    embeddings: Sequence[Sequence[float]],
    metas: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a simple in-memory vector index using lists and dot products.
    Stored as:
      { "embeddings": list[list[float]], "metadatas": list[dict] }
    """
    if len(embeddings) != len(metas):
        raise ValueError("embeddings and metas length mismatch")
    return {"embeddings": [list(map(float, e)) for e in embeddings], "metadatas": list(metas)}


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    # lightweight cosine similarity without extra deps
    # handle zero vectors defensively
    sa = sum(x * x for x in a) ** 0.5
    sb = sum(x * x for x in b) ** 0.5
    if sa == 0 or sb == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (sa * sb)


def _query_index_top_k(
    index: Dict[str, Any],
    query_embedding: Sequence[float],
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Return top_k entries as (score, metadata) ordered descending by score.
    """
    embeddings = index["embeddings"]
    metas = index["metadatas"]
    scores = []
    for emb, meta in zip(embeddings, metas):
        score = _cosine_sim(emb, query_embedding)
        scores.append((score, meta))
    scores.sort(key=lambda t: t[0], reverse=True)
    return scores[:top_k]


def rag_query(
    *,
    provider: str,
    prompt: str,
    documents: Optional[Sequence[str]] = None,
    model: str | None = None,
    embedding_model: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 5,
    timeout: float | None = 30.0,
) -> str:
    """
    High-level RAG (Retrieval-Augmented Generation) helper.

    Workflow:
    1. Validate arguments like ask(...)
    2. If documents provided: chunk them (chunk_size/overlap)
    3. Build provider and compute embeddings for chunks using provider's embed API
       (or a separate embedding_model if supported by your provider registry)
    4. Build an in-memory vector index and retrieve top-k chunks for the prompt
    5. Compose a context string containing top-k chunks and forward a combined prompt
       to the provider's generate/send method to produce a final answer.

    Notes:
    - This function expects provider instances to expose embedding methods:
      embed / embed_many / embed_batch and text generation via send(...)
    - For production-grade vector DB and FAISS, replace in-memory index with external store.
    - Returns "[ERROR] ..." on validation/runtime errors (keeps the same contract as ask).
    """
    # Basic validation reusing ask's rules where applicable
    if not isinstance(provider, str):
        return "[ERROR] provider must be string"
    provider = provider.strip().lower()
    if not provider:
        return "[ERROR] provider is empty"

    if provider not in ("auto", "echo") and provider not in PROVIDERS:
        available = ", ".join(sorted(list(PROVIDERS.keys()) + ["auto", "echo"]))
        return (
            f"[ERROR] Invalid provider '{provider}'. "
            f"Available providers: {available}"
        )

    if not isinstance(prompt, str):
        return "[ERROR] prompt must be string"
    prompt = prompt.strip()
    if not prompt:
        return "[ERROR] Invalid prompt"

    gen_provider = get_chat_provider(
                                     provider,
                                     model=model,
                                     api_key=api_key,
                                    )

    # documents may be None or sequence of strings
    docs: List[str] = []
    if documents:
        if not isinstance(documents, Sequence):
            return "[ERROR] documents must be a sequence of strings"
        for d in documents:
            if not isinstance(d, str):
                return "[ERROR] each document must be a string"
            if d.strip():
                docs.append(d.strip())

    try:
        # Build a provider for both embeddings and generation. Some providers support
        # using a different model for embeddings; delegate to registry via model/embedding_model.
        # First, try to instantiate an embedding-capable provider (embedding_model if provided)
        emb_provider = OpenAIEmbeddingProvider(
                                               model=embedding_model or "text-embedding-3-small",
                                               api_key=api_key
                                              )

        gen_provider = get_chat_provider(
                                         provider_name=provider,
                                         model=model,
                                        )

        if timeout:
            emb_provider.timeout = int(timeout)
            gen_provider.timeout = int(timeout)

        # If we have no documents, just delegate to ask/generate
        if not docs:
            return gen_provider.send(prompt)

        # Chunk documents
        passages: List[Tuple[str, Dict[str, Any]]] = []
        for i, doc in enumerate(docs):
            chunks = chunk_text(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for j, chunk in enumerate(chunks):
                meta = {"doc_index": i, "chunk_index": j}
                passages.append((chunk, meta))

        texts = [p[0] for p in passages]
        metas = [p[1] for p in passages]

        # Compute embeddings for passages
        embeddings = _compute_embeddings(emb_provider, texts)
        if not embeddings:
            return "[ERROR] Failed to compute embeddings"

        index = _build_in_memory_index(embeddings, metas)

        # Compute embedding for the query prompt
        q_embs = _compute_embeddings(emb_provider, [prompt])
        if not q_embs or not q_embs[0]:
            return "[ERROR] Failed to compute query embedding"
        query_embedding = q_embs[0]

        # Retrieve top-k relevant chunks
        top = _query_index_top_k(index, query_embedding, top_k=k)

        # Compose context from top passages
        context_parts = []
        for score, meta in top:
            # find the corresponding text from passages
            meta_idx = meta["doc_index"], meta["chunk_index"]
            # safe retrieval
            candidate_text = ""
            for t, m in passages:
                if (m["doc_index"], m["chunk_index"]) == meta_idx:
                    candidate_text = t
                    break
            context_parts.append(f"[score={score:.4f}] {candidate_text}")

        context = "\n\n".join(context_parts)
        # Compose a final prompt that includes context and user prompt
        composed_prompt = (
            "Use the following contextual passages to answer the question.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{prompt}\n\n"
            "Answer concisely and cite relevant context passages when useful."
        )

        response = gen_provider.send(composed_prompt)
        if not isinstance(response, str):
            logger.error("invalid_response_type provider=%s type=%s", provider, type(response).__name__)
            return "[ERROR] Invalid response type"
        response = response.strip()
        if not response:
            return "[ERROR] Empty response"
        return response

    except AIProviderError as exc:
        logger.error("rag_ai_provider_error provider=%s error=%s", provider, exc)
        return f"[ERROR] {exc}"
    except Exception as exc:
        logger.exception(
            "unexpected_rag_failure provider=%s model=%s error=%s",
            provider,
            model,
            exc,
        )
        return "[ERROR] Unexpected internal error. Check logs."
