import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.auto_provider import AutoProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.contracts import ChatProvider, EmbeddingProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.rag.pipeline import RAGPipeline

# ---------------------------------------------------------------------
# contracts.py
# ---------------------------------------------------------------------


class DummyChat(ChatProvider):
    def ask(self, prompt: str, **kwargs):
        return "ok"


class DummyEmbedding(EmbeddingProvider):
    def embed(self, texts, **kwargs):
        return [[0.1] for _ in texts]


def test_dummy_chat_provider():
    assert DummyChat().ask("hello") == "ok"


def test_dummy_embedding_provider():
    assert DummyEmbedding().embed(["a", "b"]) == [[0.1], [0.1]]


# ---------------------------------------------------------------------
# rag/pipeline.py
# ---------------------------------------------------------------------


def test_pipeline_fallback_when_query_missing():
    pipeline = RAGPipeline()

    pipeline.upsert_documents(["apple", "banana", "orange"])

    result = pipeline.retrieve_context("does-not-exist")

    assert result == "apple\n\nbanana\n\norange"


# ---------------------------------------------------------------------
# perplexity_provider.py
# ---------------------------------------------------------------------


def test_perplexity_ask_calls_send(monkeypatch):
    provider = object.__new__(PerplexityProvider)

    monkeypatch.setattr(
        provider,
        "send",
        lambda prompt, **kwargs: "success",
    )

    assert provider.ask("hello") == "success"


# ---------------------------------------------------------------------
# auto_provider.py
# ---------------------------------------------------------------------


class BadProvider:
    def __init__(self):
        raise RuntimeError("boom")


def test_auto_provider_init_failure(monkeypatch):
    monkeypatch.setattr(
        "ai_cli.providers.registry.ensure_initialized",
        lambda: None,
    )

    monkeypatch.setattr(
        "ai_cli.providers.registry.PROVIDER_MAP",
        {"bad": BadProvider},
        raising=False,
    )

    provider = AutoProvider(fallback_order=["bad"])

    with pytest.raises(ProviderRequestError) as exc:
        provider.send("hello")

    assert "init failed" in str(exc.value)


# ---------------------------------------------------------------------
# cohere_provider.py
# ---------------------------------------------------------------------


def test_chat_without_client_raises_runtime_error():
    provider = object.__new__(CohereProvider)

    provider.client = None
    provider.api_key = "real"

    with pytest.raises(RuntimeError):
        provider._chat("hello")


def test_embed_without_client_raises_runtime_error():
    provider = object.__new__(CohereProvider)

    provider.client = None

    with pytest.raises(RuntimeError):
        provider._embed(["hello"])


def test_embed_empty_returns_empty():
    provider = object.__new__(CohereProvider)

    assert provider._embed([]) == []


def test_upsert_documents_empty_returns():
    provider = object.__new__(CohereProvider)

    provider._documents = []
    provider._vectors = []
    provider._metadata = []

    provider.upsert_documents([])

    assert provider._documents == []
