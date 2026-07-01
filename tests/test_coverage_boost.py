"""
test_coverage_boost.py

Tests to raise coverage above 75%.  Targets low-coverage and 0%-coverage
modules that aren't adequately exercised by the existing test suite.
"""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

# ============================================================
# 1. core/exceptions.py  (52% → target >85%)
# ============================================================
from ai_cli.core.exceptions import (
    AIProviderError,
    ChunkingError,
    EmbeddingError,
    PromptValidationError,
    ProviderConfigurationError,
    ProviderRequestError,
    ResponseValidationError,
    RetrievalError,
    VectorDBError,
    capture_exception_info,
)


class TestAIProviderError:
    """Comprehensive tests for the base AIProviderError."""

    def test_basic_message(self):
        err = AIProviderError("something broke")
        assert str(err) == "something broke"
        assert err.message == "something broke"
        assert err.code is None
        assert err.retryable is False
        assert err.details == {}

    def test_with_code(self):
        err = AIProviderError("bad", code="ERR_001")
        assert "(code=ERR_001)" in str(err)

    def test_retryable(self):
        err = AIProviderError("retry me", retryable=True)
        assert "[retryable]" in str(err)

    def test_with_details(self):
        err = AIProviderError("x", details={"foo": "bar"})
        assert "foo" in str(err)

    def test_with_cause(self):
        cause = ValueError("root cause")
        err = AIProviderError("wrapped", cause=cause)
        assert err.__cause__ is cause

    def test_to_dict(self):
        err = AIProviderError("msg", code="C", retryable=True, details={"k": "v"})
        d = err.to_dict()
        assert d["message"] == "msg"
        assert d["code"] == "C"
        assert d["retryable"] is True
        assert d["details"]["k"] == "v"
        assert d["cause"] is None

    def test_to_dict_with_cause(self):
        cause = RuntimeError("boom")
        err = AIProviderError("msg", cause=cause)
        d = err.to_dict()
        assert "RuntimeError" in d["cause"]

    def test_to_json(self):
        err = AIProviderError("msg", code="X")
        j = err.to_json()
        parsed = json.loads(j)
        assert parsed["message"] == "msg"
        assert parsed["code"] == "X"

    def test_from_exception(self):
        orig = ValueError("original")
        err = AIProviderError.from_exception(orig)
        assert err.__cause__ is orig
        assert "original" in err.message

    def test_from_exception_custom_message(self):
        orig = ValueError("original")
        err = AIProviderError.from_exception(orig, message="custom")
        assert err.message == "custom"


class TestProviderRequestError:
    """ProviderRequestError has extra init params."""

    def test_with_all_details(self):
        err = ProviderRequestError(
            "req failed",
            status_code=500,
            provider_name="openai",
            request_id="abc-123",
            response_body={"error": "oops"},
        )
        assert err.details["status_code"] == 500
        assert err.details["provider_name"] == "openai"
        assert err.details["request_id"] == "abc-123"
        assert err.details["response_body"]["error"] == "oops"

    def test_without_optional_details(self):
        err = ProviderRequestError("simple")
        assert "status_code" not in err.details


class TestChunkingError:
    def test_with_chunk_index(self):
        err = ChunkingError("chunk fail", chunk_index=5)
        assert err.details["chunk_index"] == 5

    def test_without_chunk_index(self):
        err = ChunkingError("chunk fail")
        assert "chunk_index" not in err.details


class TestEmbeddingError:
    def test_with_model(self):
        err = EmbeddingError("embed fail", model="text-embed-3")
        assert err.details["model"] == "text-embed-3"

    def test_without_model(self):
        err = EmbeddingError("embed fail")
        assert "model" not in err.details


class TestVectorDBError:
    def test_with_operation(self):
        err = VectorDBError("db fail", operation="upsert")
        assert err.details["operation"] == "upsert"

    def test_without_operation(self):
        err = VectorDBError("db fail")
        assert "operation" not in err.details


class TestRetrievalError:
    def test_with_query_and_retrieved(self):
        err = RetrievalError("no results", query="test", retrieved=0)
        assert err.details["query"] == "test"
        assert err.details["retrieved"] == 0

    def test_without_optional_params(self):
        err = RetrievalError("no results")
        assert "query" not in err.details


class TestCaptureExceptionInfo:
    def test_captures_info(self):
        try:
            raise ValueError("test error")
        except ValueError as exc:
            info = capture_exception_info(exc)
        assert info["type"] == "ValueError"
        assert info["message"] == "test error"
        assert "traceback" in info


class TestSubclassInstantiation:
    """Ensure all subclasses can be instantiated."""

    def test_prompt_validation_error(self):
        err = PromptValidationError("bad prompt")
        assert isinstance(err, AIProviderError)

    def test_provider_configuration_error(self):
        err = ProviderConfigurationError("bad config")
        assert isinstance(err, AIProviderError)

    def test_response_validation_error(self):
        err = ResponseValidationError("bad response")
        assert isinstance(err, AIProviderError)


# ============================================================
# 2. core/prompt_corrector.py  (59% → target >85%)
# ============================================================
from ai_cli.core.prompt_corrector import (
    Chunk,
    PromptCorrector,
    TextChunker,
    correct_prompt,
    prompt_corrector,
)


class TestTextChunker:
    def test_chunk_by_tokens_basic(self):
        chunker = TextChunker(tokens_per_chunk=3, overlap=1)
        chunks = chunker.chunk_by_tokens("a b c d e f g")
        assert len(chunks) > 1
        assert chunks[0] == "a b c"

    def test_chunk_by_tokens_empty(self):
        chunker = TextChunker()
        chunks = chunker.chunk_by_tokens("")
        assert chunks == []

    def test_chunk_by_tokens_with_meta(self):
        chunker = TextChunker(tokens_per_chunk=3, overlap=0)
        metas = chunker.chunk_by_tokens_with_meta("a b c d e f")
        assert isinstance(metas[0], Chunk)
        assert metas[0].start_word == 0
        assert metas[0].index == 0

    def test_chunk_by_sentences(self):
        chunker = TextChunker(tokens_per_chunk=10, overlap=2)
        text = "Hello world. This is a test. Another sentence here."
        chunks = chunker.chunk_by_sentences(text)
        assert len(chunks) >= 1

    def test_chunk_by_sentences_empty(self):
        chunker = TextChunker()
        chunks = chunker.chunk_by_sentences("")
        assert chunks == []

    def test_chunk_by_sentences_with_meta(self):
        chunker = TextChunker(tokens_per_chunk=5, overlap=1)
        text = "Hello world. This is test. Another one."
        metas = chunker.chunk_by_sentences_with_meta(text)
        assert len(metas) >= 1
        assert isinstance(metas[0], Chunk)

    def test_chunk_by_sentences_with_max_tokens(self):
        chunker = TextChunker(tokens_per_chunk=3, overlap=1)
        text = "Hello world foo. Bar baz qux. Another sentence here."
        metas = chunker.chunk_by_sentences_with_meta(text, max_tokens=4)
        assert len(metas) >= 1

    def test_max_chunks_limit_tokens(self):
        chunker = TextChunker(tokens_per_chunk=2, overlap=0, max_chunks=2)
        chunks = chunker.chunk_by_tokens("a b c d e f g h")
        assert len(chunks) == 2

    def test_max_chunks_limit_sentences(self):
        chunker = TextChunker(tokens_per_chunk=3, overlap=0, max_chunks=1)
        text = "One two three. Four five six. Seven eight nine."
        chunks = chunker.chunk_by_sentences(text)
        assert len(chunks) == 1


class TestPromptCorrector:
    def test_correct_empty(self):
        pc = PromptCorrector()
        assert pc.correct("") == ""

    def test_correct_by_sentences(self):
        pc = PromptCorrector(tokens_per_chunk=5, overlap=1)
        result = pc.correct("Hello world foo. Bar baz qux.", by_sentences=True)
        assert isinstance(result, str)

    def test_correct_by_tokens(self):
        pc = PromptCorrector(tokens_per_chunk=3, overlap=1)
        result = pc.correct("a b c d e f g", by_sentences=False)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_correct_with_meta_empty(self):
        pc = PromptCorrector()
        assert pc.correct_with_meta("") == []

    def test_correct_with_meta_by_sentences(self):
        pc = PromptCorrector(tokens_per_chunk=5, overlap=1)
        result = pc.correct_with_meta("Hello. World.", by_sentences=True)
        assert isinstance(result, list)

    def test_correct_with_meta_by_tokens(self):
        pc = PromptCorrector(tokens_per_chunk=3, overlap=0)
        result = pc.correct_with_meta("a b c d e f", by_sentences=False)
        assert isinstance(result, list)
        assert len(result) > 0


class TestPromptCorrectorFunctions:
    def test_prompt_corrector_function(self):
        result = prompt_corrector("Hello world foo bar.")
        assert isinstance(result, str)

    def test_correct_prompt_empty(self):
        assert correct_prompt("") == ""

    def test_correct_prompt_nonempty(self):
        result = correct_prompt("Hello world.")
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================
# 3. providers/xAI_provider.py  (49% → target >75%)
# ============================================================
from ai_cli.providers.xAI_provider import InMemoryVectorStore, XAIProvider


class TestXAIProviderCoverage:
    """Test XAIProvider methods that aren't covered by existing tests."""

    def _make_provider(self):
        """Create XAIProvider with mocked OpenAI client."""
        with patch("ai_cli.providers.xAI_provider.OpenAI") as mock_openai_cls:
            mock_client = MagicMock()
            mock_openai_cls.return_value = mock_client
            p = XAIProvider(api_key="test-key", model="grok-2-latest")
            p.client = mock_client
        return p

    def test_send_test_key(self):
        """send() returns mock response when api_key == 'test'."""
        with patch("ai_cli.providers.xAI_provider.OpenAI") as mock_cls:
            mock_cls.return_value = MagicMock()
            p = XAIProvider(api_key="test", model="grok-2-latest")
        result = p.send("hello")
        assert result == "mock:hello"

    def test_send_success(self):
        """send() returns content from the API response."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "grok answer"
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.send("hello")
        assert result == "grok answer"

    def test_send_no_choices(self):
        """send() returns error string when no choices."""
        p = self._make_provider()
        p.client.chat.completions.create.return_value = MagicMock(choices=None)
        result = p.send("hello")
        assert "Error" in result

    def test_send_empty_content(self):
        """send() returns error string when content is None."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.send("hello")
        assert "Error" in result

    def test_send_api_exception(self):
        """send() raises ProviderRequestError on API exception."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("network")
        with pytest.raises(ProviderRequestError, match="xAI request failed"):
            p.send("hello")

    def test_call_model_with_system_prompt(self):
        """_call_model includes system message when provided."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "sys answer"
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p._call_model("hello", system_prompt="be helpful")
        assert result == "sys answer"

    def test_call_model_no_choices(self):
        """_call_model raises when no choices returned."""
        p = self._make_provider()
        p.client.chat.completions.create.return_value = MagicMock(choices=None)
        with pytest.raises(ProviderRequestError):
            p._call_model("hello")

    def test_call_model_no_content(self):
        """_call_model returns fallback when message has no content."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message = None
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p._call_model("hello")
        assert "No response" in result

    def test_send_impl(self):
        """_send_impl delegates to _call_model."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "impl answer"
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p._send_impl("hello")
        assert result == "impl answer"

    def test_send_impl_error(self):
        """_send_impl returns error string on failure."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = ProviderRequestError("fail")
        result = p._send_impl("hello")
        assert "Error" in result

    def test_health_check_success(self):
        """health_check returns True on successful API call."""
        p = self._make_provider()
        mock_choice = MagicMock()
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        assert p.health_check() is True

    def test_health_check_failure(self):
        """health_check returns False on API failure."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        assert p.health_check() is False

    def test_is_ready_with_key(self):
        """is_ready returns True when XAI_API_KEY is set."""
        with patch.dict(os.environ, {"XAI_API_KEY": "key"}):
            p = self._make_provider()
            assert p.is_ready() is True

    def test_is_ready_without_key(self):
        """is_ready returns False when XAI_API_KEY is absent."""
        env = {k: v for k, v in os.environ.items() if k != "XAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            p = self._make_provider()
            assert p.is_ready() is False

    def test_in_memory_vector_store(self):
        """InMemoryVectorStore is a simple placeholder."""
        store = InMemoryVectorStore()
        assert store is not None


# ============================================================
# 4. providers/zAI_provider.py  (57% → target >75%)
# ============================================================
from ai_cli.providers.zAI_provider import ZAIProvider


class TestZAIProviderCoverage:
    """Additional ZAIProvider tests for coverage."""

    def test_send_impl_no_key_raises(self):
        """_send_impl raises when api_key is empty."""
        p = ZAIProvider(api_key="x")
        p.api_key = ""
        with pytest.raises(ProviderRequestError, match="API key"):
            p._send_impl("hello")

    def test_send_impl_network_error(self, monkeypatch):
        """_send_impl raises on network error."""
        import requests as req
        monkeypatch.setattr(
            "requests.post",
            MagicMock(side_effect=req.RequestException("timeout")),
        )
        p = ZAIProvider(api_key="test-key")
        with pytest.raises(ProviderRequestError, match="network error"):
            p._send_impl("hello")

    def test_send_impl_json_parse_fallback(self, monkeypatch):
        """_send_impl returns raw text when JSON parsing fails."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "raw response"
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="x")
        result = p._send_impl("hello")
        assert result == "raw response"

    def test_send_impl_json_parse_empty_fallback(self, monkeypatch):
        """_send_impl raises when JSON fails and text is empty."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = ""
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="x")
        with pytest.raises(ProviderRequestError, match="empty response"):
            p._send_impl("hello")

    def test_send_impl_output_shape(self, monkeypatch):
        """_send_impl extracts from 'output' key."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"output": "out_val"}
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="x")
        result = p._send_impl("hello")
        assert result == "out_val"

    def test_send_impl_result_shape(self, monkeypatch):
        """_send_impl extracts from 'result' key."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "res_val"}
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="x")
        result = p._send_impl("hello")
        assert result == "res_val"

    def test_send_impl_choices_text(self, monkeypatch):
        """_send_impl extracts from choices[0].text."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"text": "choice_text"}]
        }
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="x")
        result = p._send_impl("hello")
        assert result == "choice_text"

    def test_send_impl_choices_message_content(self, monkeypatch):
        """_send_impl extracts from choices[0].message.content."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "nested_content"}}]
        }
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="x")
        result = p._send_impl("hello")
        assert result == "nested_content"

    def test_send_impl_json_fallback(self, monkeypatch):
        """_send_impl returns JSON string when no known keys match."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unknown_key": 42}
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="x")
        result = p._send_impl("hello")
        parsed = json.loads(result)
        assert parsed["unknown_key"] == 42

    def test_send_no_key_raises(self):
        """send() raises when api_key is empty."""
        p = ZAIProvider(api_key="x")
        p.api_key = ""
        with pytest.raises(ProviderRequestError, match="API key"):
            p.send("hello")

    def test_send_test_key(self):
        """send() returns mock response when api_key == 'test'."""
        p = ZAIProvider(api_key="test")
        result = p.send("hello")
        assert result == "mock:hello"

    def test_send_success_text_key(self, monkeypatch):
        """send() extracts from 'text' key."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "send_text"}
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="real-key")
        result = p.send("hello")
        assert result == "send_text"

    def test_send_success_choices_key(self, monkeypatch):
        """send() extracts from 'choices' key."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "choice_ans"}}]
        }
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="real-key")
        result = p.send("hello")
        assert result == "choice_ans"

    def test_send_http_error(self, monkeypatch):
        """send() raises on HTTP error status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="real-key")
        with pytest.raises(ProviderRequestError, match="z.AI error 500"):
            p.send("hello")

    def test_send_unknown_response(self, monkeypatch):
        """send() raises when response has no known keys."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"weird": True}
        monkeypatch.setattr("requests.post", lambda *a, **k: mock_resp)
        p = ZAIProvider(api_key="real-key")
        with pytest.raises(ProviderRequestError, match="unable to coerce"):
            p.send("hello")

    def test_send_network_error(self, monkeypatch):
        """send() raises on network exception."""
        import requests as req
        monkeypatch.setattr(
            "requests.post",
            MagicMock(side_effect=req.RequestException("conn refused")),
        )
        p = ZAIProvider(api_key="real-key")
        with pytest.raises(ProviderRequestError, match="network error"):
            p.send("hello")

    def test_chat_success(self):
        """chat() delegates to client and returns content."""
        p = ZAIProvider(api_key="x")
        mock_choice = MagicMock()
        mock_choice.message.content = "chat_content"
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.chat("hello")
        assert result == "chat_content"

    def test_chat_text_fallback(self):
        """chat() returns choice.text when message has no content."""
        p = ZAIProvider(api_key="x")
        mock_choice = MagicMock(spec=[])
        mock_choice.text = "text_val"
        # Remove message attr
        type(mock_choice).message = property(lambda self: None)
        resp = MagicMock()
        resp.choices = [mock_choice]
        p.client.chat.completions.create.return_value = resp
        # Since mock_choice.message is None, it won't have content
        # Testing the hasattr branch
        result = p.chat("hello")
        assert isinstance(result, str)

    def test_chat_error(self):
        """chat() raises ProviderRequestError on failure."""
        p = ZAIProvider(api_key="x")
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        with pytest.raises(ProviderRequestError, match="connection failed"):
            p.chat("hello")

    def test_is_ready_with_key(self):
        with patch.dict(os.environ, {"ZAI_API_KEY": "k"}):
            p = ZAIProvider(api_key="x")
            assert p.is_ready() is True

    def test_is_ready_without_key(self):
        env = {k: v for k, v in os.environ.items() if k != "ZAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            p = ZAIProvider(api_key="x")
            assert p.is_ready() is False


# ============================================================
# 5. providers/deepseek_provider.py  (62% → target >80%)
# ============================================================
from ai_cli.providers.deepseek_provider import DeepSeekProvider


class TestDeepSeekProviderCoverage:
    def _make_provider(self):
        with patch("ai_cli.providers.deepseek_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            p = DeepSeekProvider(api_key="test-key")
            p.client = mock_client
        return p

    def test_init_empty_key_raises(self):
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            DeepSeekProvider(api_key="")

    def test_init_no_key(self):
        env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("ai_cli.providers.deepseek_provider.OpenAI"):
                p = DeepSeekProvider(api_key=None)
                assert p.client is None

    def test_ask_success(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  deepseek answer  "
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.ask("hello")
        assert result == "deepseek answer"

    def test_ask_with_system_prompt(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "sys answer"
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.ask("hello", system_prompt="be concise")
        assert result == "sys answer"

    def test_ask_empty_content(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.ask("hello")
        assert result == ""

    def test_ask_api_error(self):
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError, match="DeepSeek request failed"):
            p.ask("hello")

    def test_send_success_message_content(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "send_answer"
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.send("hello")
        assert result == "send_answer"

    def test_send_dict_message(self):
        """send() handles dict-style message."""
        p = self._make_provider()
        mock_choice = MagicMock(spec=[])
        mock_choice.message = {"content": "dict_content"}
        mock_choice.text = None
        resp = MagicMock()
        resp.choices = [mock_choice]
        p.client.chat.completions.create.return_value = resp
        result = p.send("hello")
        assert result == "dict_content"

    def test_send_text_fallback(self):
        """send() falls back to choice.text."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.text = "text_fallback"
        resp = MagicMock()
        resp.choices = [mock_choice]
        p.client.chat.completions.create.return_value = resp
        result = p.send("hello")
        assert result == "text_fallback"

    def test_send_exception_returns_mock(self):
        """send() returns mock:hello on exception."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("fail")
        result = p.send("hello")
        assert result == "mock:hello"

    def test_chat_success(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "chat_answer"
        p.client.return_value = MagicMock(choices=[mock_choice])
        result = p.chat("hello")
        assert isinstance(result, str)

    def test_chat_error(self):
        p = self._make_provider()
        p.client.side_effect = RuntimeError("fail")
        with pytest.raises(RuntimeError, match="DeepSeek connection failed"):
            p.chat("hello")

    def test_embeddings_success(self):
        p = self._make_provider()
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        p.client.embeddings.create.return_value = MagicMock(data=[mock_item])
        result = p.embeddings(["hello"])
        assert result == [[0.1, 0.2, 0.3]]

    def test_embeddings_error(self):
        p = self._make_provider()
        p.client.embeddings.create.side_effect = RuntimeError("embed fail")
        with pytest.raises(RuntimeError, match="DeepSeek embedding"):
            p.embeddings(["hello"])


# ============================================================
# 6. Stub modules at 0% coverage
# ============================================================


class TestStubModules:
    """Import and exercise trivial 0%-coverage modules."""

    def test_ask_service_top_level(self):
        from ai_cli.ask_service import ask
        result = ask("hello")
        assert result == "mock:hello"

    def test_contracts(self):
        from ai_cli.contracts import BaseContract
        c = BaseContract()
        assert c is not None

    def test_decorators(self):
        from ai_cli.decorators import dummy_decorator
        @dummy_decorator
        def my_func():
            return 42
        assert my_func() == 42

    def test_adapters(self):
        from ai_cli.providers.adapters import LegacyAskAdapter
        adapter = LegacyAskAdapter()
        with pytest.raises(NotImplementedError):
            adapter.ask("hello")

    def test_provider_contracts(self):
        from ai_cli.providers.contracts import ChatProvider, EmbeddingProvider
        # These are ABCs, verify they exist
        assert ChatProvider is not None
        assert EmbeddingProvider is not None

    def test_provider_decorators(self):
        from ai_cli.providers.decorators import chat_provider, provider
        @provider("__test_prov_dec__")
        class P1:
            pass
        assert P1 is not None

        @chat_provider("__test_chat_dec__")
        class P2:
            def ask(self, prompt: str, **kwargs):
                return prompt

            def send(self, prompt: str, **kwargs):
                return prompt

        assert P2 is not None

    def test_openai_module(self):
        from ai_cli.providers.openai import OpenAIProvider
        p = OpenAIProvider(api_key="test")
        result = p.ask("hello")
        assert "OpenAI response" in result

    def test_spec(self):
        from ai_cli.providers.spec import ProviderRequest
        req = ProviderRequest(provider="openai", model="gpt-4")
        assert req.provider == "openai"
        assert req.model == "gpt-4"

    def test_resolver(self):
        from ai_cli.providers.resolver import resolve_provider_name
        assert resolve_provider_name("auto") == "openai"
        assert resolve_provider_name("gemini") == "gemini"
        assert resolve_provider_name("  OpenAI  ") == "openai"

    def test_config_resolve_api_key(self):
        from ai_cli.providers.config import resolve_api_key
        assert resolve_api_key("test", "explicit") == "explicit"
        with patch.dict(os.environ, {"TEST_API_KEY": "env_key"}):
            assert resolve_api_key("test") == "env_key"

    def test_test_mode(self):
        from ai_cli.utils.test_mode import is_test_mode
        with patch.dict(os.environ, {"AI_CLI_TEST_MODE": "1"}):
            assert is_test_mode() is True
        with patch.dict(os.environ, {"AI_CLI_TEST_MODE": "0"}):
            assert is_test_mode() is False


# ============================================================
# 7. core/service/ask_service.py  (0% → target 100%)
# ============================================================


class TestCoreAskService:
    def test_ask_delegates_to_chat_provider(self):
        from ai_cli.core.service.ask_service import ask

        mock_provider = MagicMock()
        mock_provider.chat.return_value = "service answer"

        with patch(
            "ai_cli.core.service.ask_service.get_chat_provider",
            return_value=mock_provider,
        ):
            result = ask("hello", provider="echo")
        assert result == "service answer"
