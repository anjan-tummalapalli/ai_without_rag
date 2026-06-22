"""
test_coverage_boost.py

Comprehensive tests targeting the lowest-coverage modules:
- ai_chat.py
- utils/secrets.py
- core/exceptions.py
- core/prompt_corrector.py
- providers/adapters.py
- providers/spec.py
- providers/openai.py  (stub provider)
- providers/decorators.py
- providers/zAI_provider.py
- providers/xAI_provider.py
- providers/deepseek_provider.py
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

# ─────────────────────────────────────────────
# ai_chat.py
# ─────────────────────────────────────────────

class TestAskFunction:
    def test_basic_ask(self):
        from ai_cli.ai_chat import ask
        result = ask("hello")
        assert "hello" in result

    def test_ask_empty_raises(self):
        from ai_cli.ai_chat import ask
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            ask("")

    def test_format_response(self):
        from ai_cli.ai_chat import format_response
        assert format_response("foo") == "foo"

    def test_chunk_text_basic(self):
        from ai_cli.ai_chat import chunk_text
        text = "hello world " * 200   # long text
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_empty(self):
        from ai_cli.ai_chat import chunk_text
        assert chunk_text("") == []

    def test_chunk_text_none(self):
        from ai_cli.ai_chat import chunk_text
        assert chunk_text(None) == []  # type: ignore[arg-type]

    def test_chunk_text_whitespace_only(self):
        from ai_cli.ai_chat import chunk_text
        assert chunk_text("   ") == []

    def test_chunk_text_short_text(self):
        from ai_cli.ai_chat import chunk_text
        result = chunk_text("short text", chunk_size=1000)
        assert result == ["short text"]

    def test_chunk_text_invalid_chunk_size(self):
        from ai_cli.ai_chat import chunk_text
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            chunk_text("hello", chunk_size=0)

    def test_chunk_text_invalid_overlap(self):
        from ai_cli.ai_chat import chunk_text
        with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
            chunk_text("hello", chunk_size=100, chunk_overlap=100)

    def test_chunk_text_no_word_split(self):
        from ai_cli.ai_chat import chunk_text
        text = "abcdefghij" * 10   # no whitespace
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5, split_on_word=False)
        assert len(chunks) >= 1

    def test_chunk_text_prefer_sentence_boundary(self):
        from ai_cli.ai_chat import chunk_text
        text = "Hello world. This is a test! Another sentence. " * 10
        chunks = chunk_text(text, chunk_size=80, chunk_overlap=10, prefer_sentence_boundary=True)
        assert len(chunks) > 1


# ─────────────────────────────────────────────
# utils/secrets.py
# ─────────────────────────────────────────────

class TestSecretsChunkText:
    def test_basic_chunk(self):
        from ai_cli.utils.secrets import chunk_text
        text = "hello world " * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1

    def test_empty_text(self):
        from ai_cli.utils.secrets import chunk_text
        assert chunk_text("") == []

    def test_short_text_fits_in_one_chunk(self):
        from ai_cli.utils.secrets import chunk_text
        assert chunk_text("tiny") == ["tiny"]

    def test_invalid_chunk_size_zero(self):
        from ai_cli.utils.secrets import chunk_text
        with pytest.raises(ValueError, match="chunk_size must be a positive int"):
            chunk_text("hello", chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        from ai_cli.utils.secrets import chunk_text
        with pytest.raises(ValueError, match="chunk_size must be a positive int"):
            chunk_text("hello", chunk_size=-1)

    def test_invalid_chunk_size_non_int(self):
        from ai_cli.utils.secrets import chunk_text
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size="big")  # type: ignore

    def test_invalid_overlap_negative(self):
        from ai_cli.utils.secrets import chunk_text
        with pytest.raises(ValueError, match="overlap must be a non-negative int"):
            chunk_text("hello", chunk_size=100, overlap=-1)

    def test_invalid_overlap_non_int(self):
        from ai_cli.utils.secrets import chunk_text
        with pytest.raises(ValueError):
            chunk_text("hello", chunk_size=100, overlap="x")  # type: ignore

    def test_overlap_equals_chunk_size(self):
        from ai_cli.utils.secrets import chunk_text
        with pytest.raises(ValueError, match="chunk_size must be greater than overlap"):
            chunk_text("hello world", chunk_size=5, overlap=5)

    def test_whitespace_normalization(self):
        from ai_cli.utils.secrets import chunk_text
        result = chunk_text("hello   world\n\t!", chunk_size=100, overlap=10)
        assert result == ["hello world !"]

    def test_no_overlap(self):
        from ai_cli.utils.secrets import chunk_text
        text = "a" * 200
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        assert len(chunks) == 4

    def test_forward_progress_guard(self):
        """Overlap close to chunk_size should not cause infinite loop."""
        from ai_cli.utils.secrets import chunk_text
        text = "x" * 100
        chunks = chunk_text(text, chunk_size=10, overlap=9)
        assert len(chunks) > 0


# ─────────────────────────────────────────────
# core/exceptions.py
# ─────────────────────────────────────────────

class TestAIProviderError:
    def test_basic_message(self):
        from ai_cli.core.exceptions import AIProviderError
        e = AIProviderError("something went wrong")
        assert str(e) == "something went wrong"

    def test_with_code(self):
        from ai_cli.core.exceptions import AIProviderError
        e = AIProviderError("err", code="E001")
        assert "E001" in str(e)

    def test_retryable_flag(self):
        from ai_cli.core.exceptions import AIProviderError
        e = AIProviderError("retry me", retryable=True)
        assert "[retryable]" in str(e)

    def test_with_details(self):
        from ai_cli.core.exceptions import AIProviderError
        e = AIProviderError("err", details={"key": "val"})
        assert "val" in str(e)

    def test_with_cause(self):
        from ai_cli.core.exceptions import AIProviderError
        cause = ValueError("original")
        e = AIProviderError("wrapped", cause=cause)
        assert e.__cause__ is cause

    def test_to_dict(self):
        from ai_cli.core.exceptions import AIProviderError
        e = AIProviderError("err", code="X", retryable=True, details={"a": 1})
        d = e.to_dict()
        assert d["message"] == "err"
        assert d["code"] == "X"
        assert d["retryable"] is True

    def test_to_json(self):
        from ai_cli.core.exceptions import AIProviderError
        e = AIProviderError("err")
        j = e.to_json()
        data = json.loads(j)
        assert data["message"] == "err"

    def test_from_exception(self):
        from ai_cli.core.exceptions import AIProviderError
        original = RuntimeError("boom")
        wrapped = AIProviderError.from_exception(original)
        assert "boom" in wrapped.message
        assert wrapped.__cause__ is original

    def test_from_exception_custom_message(self):
        from ai_cli.core.exceptions import AIProviderError
        original = RuntimeError("boom")
        wrapped = AIProviderError.from_exception(original, message="custom")
        assert wrapped.message == "custom"


class TestProviderRequestError:
    def test_status_code_in_details(self):
        from ai_cli.core.exceptions import ProviderRequestError
        e = ProviderRequestError("fail", status_code=404, provider_name="openai")
        assert e.details["status_code"] == 404
        assert e.details["provider_name"] == "openai"

    def test_request_id_in_details(self):
        from ai_cli.core.exceptions import ProviderRequestError
        e = ProviderRequestError("fail", request_id="req-123")
        assert e.details["request_id"] == "req-123"

    def test_response_body_in_details(self):
        from ai_cli.core.exceptions import ProviderRequestError
        e = ProviderRequestError("fail", response_body={"error": "rate limit"})
        assert e.details["response_body"] == {"error": "rate limit"}


class TestSpecializedExceptions:
    def test_chunking_error_with_index(self):
        from ai_cli.core.exceptions import ChunkingError
        e = ChunkingError("chunk failed", chunk_index=3)
        assert e.details["chunk_index"] == 3

    def test_embedding_error_with_model(self):
        from ai_cli.core.exceptions import EmbeddingError
        e = EmbeddingError("embed failed", model="text-embedding-3")
        assert e.details["model"] == "text-embedding-3"

    def test_vectordb_error_with_operation(self):
        from ai_cli.core.exceptions import VectorDBError
        e = VectorDBError("db failed", operation="upsert")
        assert e.details["operation"] == "upsert"

    def test_retrieval_error_with_query(self):
        from ai_cli.core.exceptions import RetrievalError
        e = RetrievalError("retrieval failed", query="what is AI?", retrieved=0)
        assert e.details["query"] == "what is AI?"
        assert e.details["retrieved"] == 0

    def test_prompt_validation_error(self):
        from ai_cli.core.exceptions import PromptValidationError
        e = PromptValidationError("bad prompt")
        assert "bad prompt" in str(e)

    def test_provider_configuration_error(self):
        from ai_cli.core.exceptions import ProviderConfigurationError
        e = ProviderConfigurationError("bad config")
        assert "bad config" in str(e)

    def test_response_validation_error(self):
        from ai_cli.core.exceptions import ResponseValidationError
        e = ResponseValidationError("bad response")
        assert "bad response" in str(e)

    def test_capture_exception_info(self):
        from ai_cli.core.exceptions import capture_exception_info
        try:
            raise ValueError("test error")
        except ValueError as exc:
            info = capture_exception_info(exc)
        assert info["type"] == "ValueError"
        assert info["message"] == "test error"
        assert "traceback" in info


# ─────────────────────────────────────────────
# core/prompt_corrector.py
# ─────────────────────────────────────────────

class TestTextChunker:
    def test_chunk_by_tokens_basic(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker(tokens_per_chunk=5, overlap=1)
        chunks = tc.chunk_by_tokens("a b c d e f g h i j")
        assert len(chunks) > 1

    def test_chunk_by_tokens_empty(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker()
        assert tc.chunk_by_tokens("") == []

    def test_chunk_by_tokens_with_meta(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker(tokens_per_chunk=3, overlap=1)
        chunks = tc.chunk_by_tokens_with_meta("one two three four five")
        assert all(hasattr(c, "text") for c in chunks)
        assert all(hasattr(c, "index") for c in chunks)

    def test_max_chunks_limit(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker(tokens_per_chunk=2, overlap=0, max_chunks=2)
        chunks = tc.chunk_by_tokens("a b c d e f g h")
        assert len(chunks) == 2

    def test_chunk_by_sentences_basic(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker(tokens_per_chunk=10, overlap=2)
        text = "Hello world. This is AI. Another sentence here."
        chunks = tc.chunk_by_sentences(text)
        assert len(chunks) >= 1

    def test_chunk_by_sentences_empty(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker()
        assert tc.chunk_by_sentences("") == []

    def test_chunk_by_sentences_with_meta(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker(tokens_per_chunk=5, overlap=1)
        text = "Short sentence. Another one! Third here?"
        chunks = tc.chunk_by_sentences_with_meta(text)
        assert isinstance(chunks, list)

    def test_chunk_by_sentences_max_chunks(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker(tokens_per_chunk=2, overlap=0, max_chunks=1)
        text = "Hello world. This is big. Another sentence."
        chunks = tc.chunk_by_sentences_with_meta(text)
        assert len(chunks) <= 1

    def test_custom_tokenizer(self):
        from ai_cli.core.prompt_corrector import TextChunker
        tc = TextChunker(tokens_per_chunk=3, tokenizer=lambda t: list(t))
        chunks = tc.chunk_by_tokens("abcdef")
        assert len(chunks) >= 1


class TestPromptCorrector:
    def test_correct_empty(self):
        from ai_cli.core.prompt_corrector import PromptCorrector
        pc = PromptCorrector()
        assert pc.correct("") == ""

    def test_correct_by_sentences(self):
        from ai_cli.core.prompt_corrector import PromptCorrector
        pc = PromptCorrector(tokens_per_chunk=20)
        result = pc.correct("Hello world. This is a test. Another sentence.")
        assert isinstance(result, str)

    def test_correct_by_tokens(self):
        from ai_cli.core.prompt_corrector import PromptCorrector
        pc = PromptCorrector(tokens_per_chunk=5)
        result = pc.correct("one two three four five six seven", by_sentences=False)
        assert isinstance(result, str)

    def test_correct_with_meta(self):
        from ai_cli.core.prompt_corrector import PromptCorrector
        pc = PromptCorrector()
        chunks = pc.correct_with_meta("Hello world.")
        assert isinstance(chunks, list)

    def test_correct_with_meta_empty(self):
        from ai_cli.core.prompt_corrector import PromptCorrector
        pc = PromptCorrector()
        assert pc.correct_with_meta("") == []

    def test_correct_with_meta_by_tokens(self):
        from ai_cli.core.prompt_corrector import PromptCorrector
        pc = PromptCorrector(tokens_per_chunk=3)
        chunks = pc.correct_with_meta("one two three four five", by_sentences=False)
        assert isinstance(chunks, list)


class TestPromptCorrectorFunctions:
    def test_prompt_corrector_fn(self):
        from ai_cli.core.prompt_corrector import prompt_corrector
        result = prompt_corrector("Hello world. This is AI.")
        assert isinstance(result, str)

    def test_correct_prompt_fn(self):
        from ai_cli.core.prompt_corrector import correct_prompt
        result = correct_prompt("Hello world.")
        assert isinstance(result, str)

    def test_correct_prompt_empty(self):
        from ai_cli.core.prompt_corrector import correct_prompt
        assert correct_prompt("") == ""


# ─────────────────────────────────────────────
# providers/adapters.py
# ─────────────────────────────────────────────

class TestLegacyAskAdapter:
    def test_ask_delegates_to_chat(self):
        from ai_cli.providers.adapters import LegacyAskAdapter

        class ConcreteAdapter(LegacyAskAdapter):
            def chat(self, prompt: str, **kwargs):
                return f"chat:{prompt}"

        adapter = ConcreteAdapter()
        result = adapter.ask("hello")
        assert result == "chat:hello"

    def test_chat_raises_not_implemented(self):
        from ai_cli.providers.adapters import LegacyAskAdapter
        adapter = LegacyAskAdapter()
        with pytest.raises(NotImplementedError, match="chat\\(\\)"):
            adapter.chat("test")

    def test_ask_calls_chat_with_kwargs(self):
        from ai_cli.providers.adapters import LegacyAskAdapter

        class ConcreteAdapter(LegacyAskAdapter):
            def chat(self, prompt: str, **kwargs):
                return f"chat:{prompt}:{kwargs.get('temperature', 'none')}"

        adapter = ConcreteAdapter()
        result = adapter.ask("hello", temperature=0.5)
        assert "0.5" in result


# ─────────────────────────────────────────────
# providers/spec.py
# ─────────────────────────────────────────────

class TestProviderRequest:
    def test_basic_creation(self):
        from ai_cli.providers.spec import ProviderRequest
        req = ProviderRequest(provider="openai")
        assert req.provider == "openai"
        assert req.model is None
        assert req.api_key is None
        assert req.kwargs is None

    def test_full_creation(self):
        from ai_cli.providers.spec import ProviderRequest
        req = ProviderRequest(provider="gemini", model="gemini-pro", api_key="key123", kwargs={"temperature": 0.7})
        assert req.provider == "gemini"
        assert req.model == "gemini-pro"
        assert req.api_key == "key123"
        assert req.kwargs == {"temperature": 0.7}

    def test_frozen_immutability(self):
        from ai_cli.providers.spec import ProviderRequest
        req = ProviderRequest(provider="openai")
        with pytest.raises((AttributeError, TypeError)):
            req.provider = "gemini"  # type: ignore[misc]

    def test_equality(self):
        from ai_cli.providers.spec import ProviderRequest
        r1 = ProviderRequest(provider="openai", model="gpt-4")
        r2 = ProviderRequest(provider="openai", model="gpt-4")
        assert r1 == r2


# ─────────────────────────────────────────────
# providers/openai.py (stub provider)
# ─────────────────────────────────────────────

class TestOpenAIStubProvider:
    def test_ask_returns_response(self):
        from ai_cli.providers.openai import OpenAIProvider
        p = OpenAIProvider(api_key="test", model="gpt-4")
        result = p.ask("hello world")
        assert "hello world" in result

    def test_init_no_args(self):
        from ai_cli.providers.openai import OpenAIProvider
        p = OpenAIProvider()
        assert p.api_key is None
        assert p.model is None


# ─────────────────────────────────────────────
# providers/zAI_provider.py
# ─────────────────────────────────────────────

class TestZAIProvider:
    def _make_provider(self, api_key="test-key"):
        from ai_cli.providers.zAI_provider import ZAIProvider
        with patch.dict("os.environ", {"ZAI_API_KEY": api_key}, clear=False):
            p = ZAIProvider(api_key=api_key)
        return p

    def test_send_mock_key(self):
        p = self._make_provider("test")
        result = p.send("hello")
        assert result == "mock:hello"

    def test_send_impl_no_api_key(self):
        from ai_cli.core.exceptions import ProviderRequestError
        from ai_cli.providers.zAI_provider import ZAIProvider
        p = ZAIProvider.__new__(ZAIProvider)
        p.api_key = ""
        p.base_url = "https://api.z.ai/v1"
        p.model = "zai-small"
        with pytest.raises(ProviderRequestError, match="API key"):
            p._send_impl("hello")

    def test_send_impl_text_field(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "hello there"}
        with patch("requests.post", return_value=mock_resp):
            result = p._send_impl("hi")
        assert result == "hello there"

    def test_send_impl_output_field(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"output": "output here"}
        with patch("requests.post", return_value=mock_resp):
            result = p._send_impl("hi")
        assert result == "output here"

    def test_send_impl_result_field(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"result": "result here"}
        with patch("requests.post", return_value=mock_resp):
            result = p._send_impl("hi")
        assert result == "result here"

    def test_send_impl_choices_text(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"text": "choice text"}]}
        with patch("requests.post", return_value=mock_resp):
            result = p._send_impl("hi")
        assert result == "choice text"

    def test_send_impl_choices_nested_message(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "nested"}}]}
        with patch("requests.post", return_value=mock_resp):
            result = p._send_impl("hi")
        assert result == "nested"

    def test_send_impl_fallback_json_dump(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"unknown_key": "unknown_value"}
        with patch("requests.post", return_value=mock_resp):
            result = p._send_impl("hi")
        assert "unknown_value" in result

    def test_send_impl_http_error(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "server error"}
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ProviderRequestError, match="500"):
                p._send_impl("hi")

    def test_send_impl_network_error(self):
        import requests as req_lib

        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider("real-key")
        with patch("requests.post", side_effect=req_lib.RequestException("timeout")):
            with pytest.raises(ProviderRequestError, match="network error"):
                p._send_impl("hi")

    def test_send_impl_non_json_response(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = "plain text response"
        with patch("requests.post", return_value=mock_resp):
            result = p._send_impl("hi")
        assert result == "plain text response"

    def test_send_impl_empty_non_json_response(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not json")
        mock_resp.text = ""
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ProviderRequestError, match="empty response"):
                p._send_impl("hi")

    def test_send_http_error(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(ProviderRequestError, match="403"):
                p.send("hi")

    def test_send_text_field(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"text": "hi back"}
        with patch("requests.post", return_value=mock_resp):
            result = p.send("hi")
        assert result == "hi back"

    def test_send_choices_field(self):
        p = self._make_provider("real-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "choice content"}}]}
        with patch("requests.post", return_value=mock_resp):
            result = p.send("hi")
        assert result == "choice content"

    def test_send_network_error(self):
        import requests as req_lib

        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider("real-key")
        with patch("requests.post", side_effect=req_lib.RequestException("conn refused")):
            with pytest.raises(ProviderRequestError, match="network error"):
                p.send("hi")

    def test_send_no_api_key_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        from ai_cli.providers.zAI_provider import ZAIProvider
        p = ZAIProvider.__new__(ZAIProvider)
        p.api_key = ""
        p.base_url = "https://api.z.ai/v1"
        p.model = "zai-small"
        with pytest.raises(ProviderRequestError):
            p.send("hi")


# ─────────────────────────────────────────────
# providers/xAI_provider.py
# ─────────────────────────────────────────────

class TestXAIProvider:
    def _make_provider(self, api_key="test"):
        from ai_cli.providers.xAI_provider import XAIProvider
        mock_client = MagicMock()
        with patch("ai_cli.providers.xAI_provider.OpenAI", return_value=mock_client):
            p = XAIProvider(model="grok-2", api_key=api_key)
            p.client = mock_client
        return p

    def test_send_mock_key(self):
        p = self._make_provider("test")
        result = p.send("hello")
        assert result == "mock:hello"

    def test_send_success(self):
        p = self._make_provider("real-key")
        mock_choice = MagicMock()
        mock_choice.message.content = "grok response"
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.send("hello")
        assert result == "grok response"

    def test_send_no_choices(self):
        p = self._make_provider("real-key")
        p.client.chat.completions.create.return_value = MagicMock(choices=[])
        result = p.send("hello")
        assert "[Error" in result

    def test_send_no_content(self):
        p = self._make_provider("real-key")
        mock_choice = MagicMock()
        mock_choice.message.content = None
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.send("hello")
        assert "[Error" in result

    def test_send_raises_provider_error(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider("real-key")
        p.client.chat.completions.create.side_effect = RuntimeError("API down")
        with pytest.raises(ProviderRequestError, match="xAI request failed"):
            p.send("hello")

    def test_call_model_no_choices(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider("real-key")
        p.client.chat.completions.create.return_value = MagicMock(choices=None)
        with pytest.raises(ProviderRequestError, match="no completion choices"):
            p._call_model("hello")

    def test_call_model_empty_content(self):
        p = self._make_provider("real-key")
        mock_choice = MagicMock()
        mock_choice.message.content = None
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p._call_model("hello")
        assert result == "[No response from xAI]"

    def test_call_model_with_system_prompt(self):
        p = self._make_provider("real-key")
        mock_choice = MagicMock()
        mock_choice.message.content = "  response  "
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p._call_model("hello", system_prompt="You are helpful")
        assert result == "response"

    def test_send_impl_wraps_error(self):
        p = self._make_provider("real-key")
        from ai_cli.core.exceptions import ProviderRequestError
        p.client.chat.completions.create.side_effect = ProviderRequestError("fail")
        result = p._send_impl("hello")
        assert "[Error" in result

    def test_health_check_success(self):
        p = self._make_provider("real-key")
        mock_choice = MagicMock()
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.health_check() is True

    def test_health_check_failure(self):
        p = self._make_provider("real-key")
        p.client.chat.completions.create.side_effect = RuntimeError("network error")
        assert p.health_check() is False

    def test_in_memory_vector_store_import(self):
        from ai_cli.providers.xAI_provider import InMemoryVectorStore
        store = InMemoryVectorStore()
        assert store is not None


# ─────────────────────────────────────────────
# providers/deepseek_provider.py
# ─────────────────────────────────────────────

class TestDeepSeekProvider:
    def _make_provider(self, api_key="test-key"):
        with patch("ai_cli.providers.deepseek_provider.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            from ai_cli.providers.deepseek_provider import DeepSeekProvider
            p = DeepSeekProvider(api_key=api_key)
            p.client = mock_client
        return p

    def test_init_missing_key(self):
        import os

        from ai_cli.providers.deepseek_provider import DeepSeekProvider
        with patch.dict(os.environ, {}, clear=True):
            # Remove DEEPSEEK_API_KEY from environment
            env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                    DeepSeekProvider(api_key=None)

    def test_provider_name(self):
        p = self._make_provider()
        assert p.provider_name == "deepseek"

    def test_ask_success(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  hello  "
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.ask("test prompt")
        assert result == "hello"

    def test_ask_empty_content(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = None
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.ask("test")
        assert result == ""

    def test_ask_with_system_prompt(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.ask("test", system_prompt="be helpful")
        assert result == "response"

    def test_ask_failure_raises(self):
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("API error")
        with pytest.raises(RuntimeError, match="DeepSeek request failed"):
            p.ask("test")

    def test_embeddings_success(self):
        p = self._make_provider()
        mock_item1 = MagicMock()
        mock_item1.embedding = [0.1, 0.2, 0.3]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.4, 0.5, 0.6]
        p.client.embeddings.create.return_value = MagicMock(data=[mock_item1, mock_item2])
        result = p.embeddings(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    def test_embeddings_failure(self):
        p = self._make_provider()
        p.client.embeddings.create.side_effect = RuntimeError("embed error")
        with pytest.raises(RuntimeError, match="DeepSeek embedding request failed"):
            p.embeddings(["text"])

    def test_health_check_true(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.health_check() is True

    def test_health_check_false(self):
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        assert p.health_check() is False

    def test_send_with_choices(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "sent!"
        mock_choice.text = None
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice], spec=["choices"]
        )
        result = p.send("hello")
        assert result == "sent!"

    def test_send_fallback_string(self):
        p = self._make_provider()
        p.client.chat.completions.create.return_value = "raw string"
        result = p.send("hello")
        # Falls through to isinstance(response, str)
        assert "raw string" in result or result == "mock:hello"
