from unittest.mock import Mock

import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.gemini_provider import GeminiProvider


def make_provider():
    p = GeminiProvider(api_key="test-key")
    return p


def test_embed_new_sdk_missing_models():
    p = make_provider()
    p.client = object()

    with pytest.raises(ProviderRequestError):
        p._embed_with_new_sdk("model", ["hello"])


def test_embed_new_sdk_empty_embeddings():
    p = make_provider()

    result = Mock()
    result.embeddings = []

    models = Mock()
    models.embed_content.return_value = result

    p.client = Mock()
    p.client.models = models

    with pytest.raises(ProviderRequestError):
        p._embed_with_new_sdk("model", ["hello"])


def test_embed_new_sdk_missing_values():
    p = make_provider()

    emb = Mock()
    emb.values = None

    result = Mock()
    result.embeddings = [emb]

    models = Mock()
    models.embed_content.return_value = result

    p.client = Mock()
    p.client.models = models

    with pytest.raises(ProviderRequestError):
        p._embed_with_new_sdk("model", ["hello"])


def test_create_embeddings_returns_none(monkeypatch):
    p = make_provider()

    monkeypatch.setattr(
        p,
        "_embed_with_new_sdk",
        lambda *a, **k: [],
    )

    with pytest.raises(ProviderRequestError):
        p._create_embeddings(["hello"])


def test_create_embeddings_exception(monkeypatch):
    p = make_provider()

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        p,
        "_embed_with_new_sdk",
        boom,
    )

    with pytest.raises(ProviderRequestError):
        p._create_embeddings(["hello"])


def test_send_calls_send_impl(monkeypatch):
    p = make_provider()

    called = {}

    def fake(prompt):
        called["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(
        p,
        "_send_impl",
        fake,
    )

    assert p.send("hello") == "ok"
    assert called["prompt"] == "hello"


def test_is_ready_false(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    p = make_provider()

    assert p.is_ready() is False


def test_is_ready_true(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "abc")

    p = make_provider()

    assert p.is_ready() is True


def test_query_vector_db_empty_embeddings(monkeypatch):
    p = make_provider()

    monkeypatch.setattr(
        p,
        "_create_embeddings",
        lambda *args, **kwargs: [],
    )

    with pytest.raises(ProviderRequestError):
        p.query_vector_db("hello")


def test_query_vector_db_embedding_failure(monkeypatch):
    p = make_provider()

    def boom(*args, **kwargs):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(
        p,
        "_create_embeddings",
        boom,
    )

    with pytest.raises(ProviderRequestError):
        p.query_vector_db("hello")


def test_query_vector_db_vector_store_failure(monkeypatch):
    p = make_provider()

    monkeypatch.setattr(
        p,
        "_create_embeddings",
        lambda *args, **kwargs: [[1.0, 2.0]],
    )

    class BadVectorDB:
        def query(self, *args, **kwargs):
            raise RuntimeError("database failed")

    p.vector_db = BadVectorDB()

    with pytest.raises(ProviderRequestError):
        p.query_vector_db("hello")
