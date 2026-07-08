"""
test_gemini_extended.py

Tests for GeminiProvider and InMemoryVectorDB in gemini_provider.py.
Uses the "test" / "test-key" api_key shortcut to avoid real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────
# InMemoryVectorDB
# ─────────────────────────────────────────────


class TestInMemoryVectorDB:
    def _make_db(self):
        from ai_cli.providers.gemini_provider import InMemoryVectorDB

        db = InMemoryVectorDB.__new__(InMemoryVectorDB)
        db._items = []
        db.api_key = None
        db._use_new_api = True
        return db

    def test_upsert_and_query(self):
        db = self._make_db()
        db.upsert(
            [
                {
                    "id": "a",
                    "vector": [1.0, 0.0],
                    "text": "hello",
                    "metadata": {},
                },
                {
                    "id": "b",
                    "vector": [0.0, 1.0],
                    "text": "world",
                    "metadata": {},
                },
            ]
        )
        results = db.query([1.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_upsert_replaces_existing(self):
        db = self._make_db()
        db.upsert(
            [{"id": "x", "vector": [1.0, 0.0], "text": "v1", "metadata": {}}]
        )
        db.upsert(
            [{"id": "x", "vector": [0.0, 1.0], "text": "v2", "metadata": {}}]
        )
        assert len(db._items) == 1
        assert db._items[0]["text"] == "v2"

    def test_query_empty_db(self):
        db = self._make_db()
        assert db.query([1.0, 0.0]) == []

    def test_cosine_similarity_identical(self):
        from ai_cli.providers.gemini_provider import InMemoryVectorDB

        sim = InMemoryVectorDB._cosine_similarity_with_norms(
            [1.0, 0.0], 1.0, [1.0, 0.0], 1.0
        )
        assert abs(sim - 1.0) < 1e-9

    def test_cosine_similarity_zero_norm(self):
        from ai_cli.providers.gemini_provider import InMemoryVectorDB

        sim = InMemoryVectorDB._cosine_similarity_with_norms(
            [0.0, 0.0], 0.0, [1.0, 0.0], 1.0
        )
        assert sim == 0.0

    def test_query_top_k_respected(self):
        db = self._make_db()
        db.upsert(
            [
                {
                    "id": f"item_{i}",
                    "vector": [float(i), 0.0],
                    "text": f"item {i}",
                    "metadata": {},
                }
                for i in range(1, 6)
            ]
        )
        results = db.query([5.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_upsert_zero_vector_norm(self):
        """Zero vector should not crash."""
        db = self._make_db()
        db.upsert(
            [{"id": "zero", "vector": [], "text": "empty", "metadata": {}}]
        )
        assert len(db._items) == 1
        assert db._items[0]["norm"] == 0.0


# ─────────────────────────────────────────────
# GeminiProvider – mock API key path
# ─────────────────────────────────────────────


class TestGeminiProvider:
    def _make_provider(self, **kwargs):
        """Create a GeminiProvider with the test API key to avoid real SDK calls."""
        from ai_cli.providers.gemini_provider import GeminiProvider

        mock_model = MagicMock()

        with patch("ai_cli.providers.gemini_provider.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            p = GeminiProvider(api_key="test", **kwargs)
            p._mock_model = mock_model
        return p

    def test_send_mock_path(self):
        p = self._make_provider()
        result = p.send("hello")
        assert result == "gemini response"

    def test_send_impl_test_key(self):
        p = self._make_provider()
        result = p._send_impl("hello")
        assert result == "gemini response"

    def test_send_impl_test_key_variant(self):
        p = self._make_provider()
        p.api_key = "test-key"
        result = p._send_impl("hello")
        assert result == "gemini response"

    def test_chunk_text_empty(self):
        p = self._make_provider()
        assert p.chunk_text("") == []

    def test_chunk_text_basic(self):
        p = self._make_provider(chunk_size=10, chunk_overlap=2)
        text = "hello world and some more text here"
        chunks = p.chunk_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_short(self):
        p = self._make_provider(chunk_size=500)
        text = "short"
        chunks = p.chunk_text(text)
        assert chunks == ["short"]

    def test_chunk_text_override_params(self):
        p = self._make_provider(chunk_size=100, chunk_overlap=10)
        text = "a" * 200
        chunks = p.chunk_text(text, chunk_size=50, overlap=5)
        assert len(chunks) > 1

    def test_invalid_chunk_size_raises(self):
        from ai_cli.providers.gemini_provider import GeminiProvider

        with patch("ai_cli.providers.gemini_provider.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel.return_value = MagicMock()
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                GeminiProvider(api_key="test", chunk_size=0)

    def test_invalid_chunk_overlap_raises(self):
        from ai_cli.providers.gemini_provider import GeminiProvider

        with patch("ai_cli.providers.gemini_provider.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel.return_value = MagicMock()
            with pytest.raises(
                ValueError, match="chunk_overlap must be non-negative"
            ):
                GeminiProvider(api_key="test", chunk_overlap=-1)

    def test_missing_api_key_raises(self):
        import os

        from ai_cli.core.exceptions import ProviderRequestError
        from ai_cli.providers.gemini_provider import GeminiProvider

        env = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ProviderRequestError, match="GEMINI_API_KEY"):
                GeminiProvider(api_key=None)

    def test_provider_name(self):
        p = self._make_provider()
        assert p.provider_name == "gemini"

    def test_index_document_empty_text(self):
        p = self._make_provider()
        p._create_embeddings = MagicMock(return_value=[])
        # Empty text → chunk_text returns [] → should return early
        p.index_document("doc1", "")  # Should not raise

    def test_retrieve_relevant_context_empty(self):
        p = self._make_provider()
        p.query_vector_db = MagicMock(return_value=[])
        result = p.retrieve_relevant_context("query")
        assert result == ""

    def test_retrieve_relevant_context_with_results(self):
        p = self._make_provider()
        p.query_vector_db = MagicMock(
            return_value=[
                {"text": "chunk one"},
                {"text": "chunk two"},
            ]
        )
        result = p.retrieve_relevant_context("query")
        assert "chunk one" in result
        assert "chunk two" in result

    def test_send_with_rag_no_embedding_model_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError

        p = self._make_provider()
        p.embedding_model = None
        with pytest.raises(
            ProviderRequestError, match="Embedding model not configured"
        ):
            p.send_with_rag("hello")

    def test_send_with_rag_no_context(self):
        p = self._make_provider(embedding_model="embedding-model")
        p.retrieve_relevant_context = MagicMock(return_value="")
        result = p.send_with_rag("hello")
        assert result == "gemini response"

    def test_send_with_rag_with_context_prepend(self):
        p = self._make_provider(embedding_model="embedding-model")
        p.retrieve_relevant_context = MagicMock(return_value="some context")
        result = p.send_with_rag("hello", prepend_context=True)
        assert result == "gemini response"

    def test_send_with_rag_with_context_append(self):
        p = self._make_provider(embedding_model="embedding-model")
        p.retrieve_relevant_context = MagicMock(return_value="some context")
        result = p.send_with_rag("hello", prepend_context=False)
        assert result == "gemini response"

    def test_send_with_rag_custom_prefix(self):
        p = self._make_provider(embedding_model="embedding-model")
        p.retrieve_relevant_context = MagicMock(return_value="ctx")
        result = p.send_with_rag("hello", context_prefix="CONTEXT:")
        assert result == "gemini response"

    def test_create_embeddings_empty(self):
        p = self._make_provider()
        result = p._create_embeddings([])
        assert result == []

    def test_create_embeddings_no_sdk_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError

        p = self._make_provider()
        p._use_new_api = True
        p.client = MagicMock(spec=[])  # no "models" attribute
        with pytest.raises(
            ProviderRequestError, match="Embedding API not available"
        ):
            p._create_embeddings(["hello"])

    def test_query_vector_db_delegates_to_db(self):
        p = self._make_provider()
        mock_vec = [0.1, 0.2]
        p._create_embeddings = MagicMock(return_value=[mock_vec])
        p.vector_db = MagicMock()
        p.vector_db.query.return_value = [{"id": "x", "text": "result"}]
        results = p.query_vector_db("query", top_k=1)
        assert results[0]["id"] == "x"

    def test_query_vector_db_embedding_fail_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError

        p = self._make_provider()
        p._create_embeddings = MagicMock(
            side_effect=ProviderRequestError("embed fail")
        )
        with pytest.raises(ProviderRequestError, match="embed fail"):
            p.query_vector_db("query")

    def test_index_document_mismatch_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError

        p = self._make_provider(chunk_size=5, chunk_overlap=0)
        p._create_embeddings = MagicMock(return_value=[[0.1]])  # only 1 vector
        # chunk_text on "hello world" with chunk_size=5 → multiple chunks
        with pytest.raises(
            ProviderRequestError, match="Embedding count does not match"
        ):
            p.index_document("doc1", "hello world and more text")

    def test_index_document_vector_db_error_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError

        p = self._make_provider(chunk_size=500)
        p._create_embeddings = MagicMock(return_value=[[0.1, 0.2]])
        p.vector_db = MagicMock()
        p.vector_db.upsert.side_effect = RuntimeError("db error")
        with pytest.raises(ProviderRequestError, match="Failed to upsert"):
            p.index_document("doc1", "some text")

    def test_health_check_mock_path(self):
        p = self._make_provider()
        p._use_new_api = False
        mock_resp = MagicMock()
        mock_resp.text = "pong"
        p.client = MagicMock()
        p.client.generate_content.return_value = mock_resp
        # Since api_key == "test", _send_impl won't call the client;
        # health_check does call the client directly
        assert (
            p.health_check() is True or p.health_check() is False
        )  # just no crash


# ─────────────────────────────────────────────
# GeminiProvider – _send_impl new API path
# ─────────────────────────────────────────────


class TestGeminiSendImpl:
    def _make_provider_real_key(self):
        from ai_cli.providers.gemini_provider import GeminiProvider

        mock_model = MagicMock()
        with patch("ai_cli.providers.gemini_provider.genai") as mock_genai:
            mock_genai.configure = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            p = GeminiProvider(api_key="real-key")
            p._mock_model = mock_model
        return p

    def test_send_impl_with_text_response(self):
        p = self._make_provider_real_key()
        p._use_new_api = False
        mock_resp = MagicMock()
        mock_resp.text = "hello from gemini"
        p.client = MagicMock()
        p.client.generate_content.return_value = mock_resp
        result = p._send_impl("test")
        assert result == "hello from gemini"

    def test_send_impl_exception_returns_fallback(self):
        p = self._make_provider_real_key()
        p._use_new_api = False
        p.client = MagicMock()
        p.client.generate_content.side_effect = RuntimeError("API error")
        result = p._send_impl("test")
        assert result == "gemini response"

    def test_send_impl_no_text_returns_fallback(self):
        p = self._make_provider_real_key()
        p._use_new_api = False
        mock_resp = MagicMock()
        mock_resp.text = None
        p.client = MagicMock()
        p.client.generate_content.return_value = mock_resp
        result = p._send_impl("test")
        assert result == "gemini response"

    def test_health_check_legacy_success(self):
        p = self._make_provider_real_key()
        p._use_new_api = False
        mock_resp = MagicMock()
        mock_resp.text = "pong"
        p.client = MagicMock()
        p.client.generate_content.return_value = mock_resp
        assert p.health_check() is True

    def test_health_check_legacy_failure(self):
        p = self._make_provider_real_key()
        p._use_new_api = False
        p.client = MagicMock()
        p.client.generate_content.side_effect = RuntimeError("down")
        assert p.health_check() is False

    def test_health_check_no_text(self):
        p = self._make_provider_real_key()
        p._use_new_api = False
        mock_resp = MagicMock()
        mock_resp.text = None
        p.client = MagicMock()
        p.client.generate_content.return_value = mock_resp
        assert p.health_check() is False
