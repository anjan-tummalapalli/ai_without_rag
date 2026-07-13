import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.gemini_provider import GeminiProvider


def provider():
    return GeminiProvider(
        api_key="test-key",
        embedding_model="embedding-model",
    )


# ------------------------------------------------------------------
# _send_impl
# ------------------------------------------------------------------


def test_send_impl_test_key():
    p = provider()
    assert p._send_impl("hello") == "gemini response"


# ------------------------------------------------------------------
# health_check
# ------------------------------------------------------------------


def test_health_check_true():
    p = provider()

    class Response:
        text = "pong"

    class Models:
        def generate_content(self, **kwargs):
            return Response()

    class Client:
        models = Models()

    p._use_new_api = True
    p.client = Client()

    assert p.health_check() is True


def test_health_check_false():
    p = provider()

    class Models:
        def generate_content(self, **kwargs):
            raise RuntimeError()

    class Client:
        models = Models()

    p._use_new_api = True
    p.client = Client()

    assert p.health_check() is False


# ------------------------------------------------------------------
# chunk_text
# ------------------------------------------------------------------


def test_chunk_text_empty():
    p = provider()
    assert p.chunk_text("") == []


def test_chunk_text_normal():
    p = provider()

    chunks = p.chunk_text(
        "abcdefghijklmnopqrstuvwxyz",
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) >= 3
    assert chunks[0] == "abcdefghij"


# ------------------------------------------------------------------
# index_document
# ------------------------------------------------------------------


def test_index_document_no_chunks(monkeypatch):
    p = provider()

    monkeypatch.setattr(
        p,
        "chunk_text",
        lambda text: [],
    )

    p.index_document("1", "hello")


def test_index_document_embedding_mismatch(monkeypatch):
    p = provider()

    monkeypatch.setattr(
        p,
        "chunk_text",
        lambda text: ["a", "b"],
    )

    monkeypatch.setattr(
        p,
        "_create_embeddings",
        lambda chunks: [[1.0]],
    )

    with pytest.raises(ProviderRequestError):
        p.index_document("1", "hello")


def test_index_document_upsert_failure(monkeypatch):
    p = provider()

    monkeypatch.setattr(
        p,
        "chunk_text",
        lambda text: ["abc"],
    )

    monkeypatch.setattr(
        p,
        "_create_embeddings",
        lambda chunks: [[1.0, 2.0]],
    )

    class BadDB:
        def upsert(self, items):
            raise RuntimeError("boom")

    p.vector_db = BadDB()

    with pytest.raises(ProviderRequestError):
        p.index_document("1", "hello")


# ------------------------------------------------------------------
# retrieve_relevant_context
# ------------------------------------------------------------------


def test_retrieve_context_empty(monkeypatch):
    p = provider()

    monkeypatch.setattr(
        p,
        "query_vector_db",
        lambda q, top_k=3: [],
    )

    assert p.retrieve_relevant_context("hello") == ""


def test_retrieve_context_success(monkeypatch):
    p = provider()

    monkeypatch.setattr(
        p,
        "query_vector_db",
        lambda q, top_k=3: [
            {"text": "abc"},
            {"text": "xyz"},
        ],
    )

    result = p.retrieve_relevant_context("hello")

    assert "abc" in result
    assert "xyz" in result
    assert "---" in result


# ------------------------------------------------------------------
# send_with_rag
# ------------------------------------------------------------------


def test_send_with_rag_missing_embedding():
    p = GeminiProvider(api_key="test-key")

    with pytest.raises(ProviderRequestError):
        p.send_with_rag("hello")


def test_send_with_rag_with_context(monkeypatch):
    p = provider()

    monkeypatch.setattr(
        p,
        "retrieve_relevant_context",
        lambda *a, **k: "context",
    )

    monkeypatch.setattr(
        p,
        "_send_impl",
        lambda prompt: prompt,
    )

    result = p.send_with_rag("hello")

    assert "context" in result
    assert "hello" in result


def test_send_with_rag_without_context(monkeypatch):
    p = provider()

    monkeypatch.setattr(
        p,
        "retrieve_relevant_context",
        lambda *a, **k: "",
    )

    monkeypatch.setattr(
        p,
        "_send_impl",
        lambda prompt: prompt,
    )

    assert p.send_with_rag("hello") == "hello"
