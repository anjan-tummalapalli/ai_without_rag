import sys
from unittest.mock import MagicMock

import pytest

try:
    from ai_cli.providers.gemini_provider import (
        GeminiProvider,
        InMemoryVectorDB,
        ProviderRequestError,
        genai,
    )
except Exception as exc:
    GeminiProvider = None
    ProviderRequestError = RuntimeError
    InMemoryVectorDB = None
    genai = None
    GEMINI_IMPORT_ERROR = exc
else:
    GEMINI_IMPORT_ERROR = None


pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="Gemini SDK is incompatible with Python 3.14 in this environment",
)


def test_gemini_import_available():
    if GeminiProvider is None:
        pytest.skip(f"Gemini unavailable: {GEMINI_IMPORT_ERROR}")
    assert GeminiProvider is not None


@pytest.mark.skipif(
    GeminiProvider is None, reason="Gemini provider unavailable"
)
class TestGeminiCoverageBoost:
    def _provider(self):
        p = GeminiProvider.__new__(GeminiProvider)
        p.model = "gemini"
        p.api_key = "test"
        p.embedding_model = "embed"
        p.chunk_size = 10
        p.chunk_overlap = 2
        p._mock = False
        p._use_new_api = False
        p.vector_db = MagicMock()
        p.client = MagicMock()
        return p

    def test_cosine_zero_norm(self):
        assert (
            InMemoryVectorDB._cosine_similarity_with_norms([1], 0.0, [1], 1.0)
            == 0.0
        )

    def test_cosine_normal(self):
        score = InMemoryVectorDB._cosine_similarity_with_norms(
            [1.0, 0.0],
            1.0,
            [1.0, 0.0],
            1.0,
        )
        assert score == pytest.approx(1.0)

    def test_query_empty(self):
        db = InMemoryVectorDB.__new__(InMemoryVectorDB)
        db._items = []
        assert db.query([1]) == []

    def test_query_all_results(self):
        db = InMemoryVectorDB.__new__(InMemoryVectorDB)
        db._items = [
            {
                "id": "1",
                "vector": [1.0],
                "norm": 1.0,
                "metadata": {},
                "text": "abc",
            }
        ]
        result = db.query([1.0], top_k=5)
        assert result[0]["id"] == "1"

    def test_query_heap_branch(self):
        db = InMemoryVectorDB.__new__(InMemoryVectorDB)
        db._items = [
            {
                "id": str(i),
                "vector": [float(i + 1)],
                "norm": float(i + 1),
                "metadata": {},
                "text": str(i),
            }
            for i in range(5)
        ]
        result = db.query([1.0], top_k=2)
        assert len(result) == 2

    def test_chunk_empty(self):
        p = self._provider()
        assert p.chunk_text("") == []

    def test_chunk_single(self):
        p = self._provider()
        assert p.chunk_text("hello") == ["hello"]

    def test_chunk_multiple(self):
        p = self._provider()
        chunks = p.chunk_text("abcdefghijklmnopqrstuvwxyz")
        assert len(chunks) > 1

    def test_embeddings_empty(self):
        p = self._provider()
        assert p._create_embeddings([]) == []

    def test_embeddings_api_missing(self, monkeypatch):
        p = self._provider()

        monkeypatch.delattr(genai, "embed_content", raising=False)

        with pytest.raises(ProviderRequestError):
            p._create_embeddings(["abc"])

    def test_embeddings_missing_vector(self, monkeypatch):
        monkeypatch.setattr(
            genai, "embed_content", lambda **kwargs: {}, raising=False
        )

        p = self._provider()

        with pytest.raises(ProviderRequestError):
            p._create_embeddings(["abc"])

    def test_embeddings_exception(self, monkeypatch):
        def _raise(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(genai, "embed_content", _raise, raising=False)

        p = self._provider()

        with pytest.raises(ProviderRequestError):
            p._create_embeddings(["abc"])

    def test_index_document_no_chunks(self):
        p = self._provider()
        p.chunk_text = MagicMock(return_value=[])
        p.index_document("doc", "text")
        p.vector_db.upsert.assert_not_called()

    def test_index_document_count_mismatch(self):
        p = self._provider()

        p.chunk_text = MagicMock(return_value=["a", "b"])
        p._create_embeddings = MagicMock(return_value=[[1]])

        with pytest.raises(ProviderRequestError):
            p.index_document("doc", "text")

    def test_index_document_upsert_exception(self):
        p = self._provider()

        p.chunk_text = MagicMock(return_value=["a"])
        p._create_embeddings = MagicMock(return_value=[[1]])
        p.vector_db.upsert.side_effect = RuntimeError("boom")

        with pytest.raises(ProviderRequestError):
            p.index_document("doc", "text")

    def test_query_embedding_failure(self):
        p = self._provider()
        p._create_embeddings = MagicMock(return_value=[])
        with pytest.raises(ProviderRequestError):
            p.query_vector_db("abc")

    def test_query_vector_db_exception(self):
        p = self._provider()
        p._create_embeddings = MagicMock(return_value=[[1]])
        p.vector_db.query.side_effect = RuntimeError("boom")

        with pytest.raises(ProviderRequestError):
            p.query_vector_db("abc")

    def test_context_none(self):
        p = self._provider()
        p.query_vector_db = MagicMock(return_value=[])
        assert p.retrieve_relevant_context("abc") == ""

    def test_context_join(self):
        p = self._provider()
        p.query_vector_db = MagicMock(
            return_value=[
                {"text": "one"},
                {"text": "two"},
            ]
        )
        result = p.retrieve_relevant_context("abc")
        assert "one" in result
        assert "two" in result

    def test_send_with_rag_no_embedding_model(self):
        p = self._provider()
        p.embedding_model = None
        with pytest.raises(ProviderRequestError):
            p.send_with_rag("hello")

    def test_send_with_rag_no_context(self):
        p = self._provider()
        p.retrieve_relevant_context = MagicMock(return_value="")
        p._send_impl = MagicMock(return_value="ok")
        assert p.send_with_rag("hello") == "ok"

    def test_send_with_rag_append_context(self):
        p = self._provider()
        p.retrieve_relevant_context = MagicMock(return_value="CTX")
        p._send_impl = MagicMock(return_value="ok")
        p.send_with_rag("hello", prepend_context=False)
        sent = p._send_impl.call_args.args[0]
        assert sent.startswith("hello")
        assert "CTX" in sent

    def test_send_mock(self):
        p = self._provider()
        p._mock = True
        assert p.send("abc") == "gemini response"

    def test_health_exception(self):
        p = self._provider()
        p.client.generate_content.side_effect = RuntimeError()
        assert p.health_check() is False

    def test_health_dict_response(self):
        p = self._provider()
        p.client.generate_content.return_value = {"text": "pong"}
        assert p.health_check() is True

    def test_health_empty(self):
        p = self._provider()
        p.client.generate_content.return_value = {}
        assert p.health_check() is False
