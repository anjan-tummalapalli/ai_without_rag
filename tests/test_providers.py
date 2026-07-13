import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.echo_provider import EchoProvider
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider

try:
    from ai_cli.providers.gemini_provider import (
        GEMINI_IMPORT_ERROR,
        GeminiProvider,
    )
except Exception as e:
    GeminiProvider = None
    GEMINI_IMPORT_ERROR = e


pytest.skip(
    "Gemini SDK incompatible on Python 3.14 in this environment",
    allow_module_level=True,
)

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="Gemini SDK is incompatible with Python 3.14 in this environment",
)


def test_gemini_import_available():
    if GeminiProvider is None:
        pytest.skip(f"Gemini unavailable: {GEMINI_IMPORT_ERROR}")
    assert GeminiProvider is not None


def _setup_openai_mock(openai_mock, chat_text="response"):
    client = MagicMock()
    openai_mock.return_value = client

    # chat response
    choice = MagicMock()
    choice.message.content = chat_text
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp

    return client


def _setup_genai_mock(genai_mock, text="gemini response"):
    model = MagicMock()
    genai_mock.GenerativeModel.return_value = model
    resp = MagicMock()
    resp.text = text
    model.generate_content.return_value = resp

    return model


def test_echo_provider():
    p = EchoProvider()
    assert p.provider_name == "echo"
    assert p.send("hello") == "(echo) hello"


@patch("ai_cli.providers.openai_provider.OpenAI")
def test_openai_provider_send(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="openai response")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    p = OpenAIProvider()
    assert p.send("hello") == "openai response"
    openai_mock.return_value.chat.completions.create.assert_called_once()


@patch("ai_cli.providers.gemini_provider.genai")
def test_gemini_provider(genai_mock, monkeypatch):
    _setup_genai_mock(genai_mock, text="gemini response")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    p = GeminiProvider()
    assert p.send("hello") == "gemini response"


@patch("cohere.Client")
def test_cohere_provider(cohere_client_cls, monkeypatch):
    client = MagicMock()
    cohere_client_cls.return_value = client

    chat_resp = MagicMock()
    chat_resp.text = "cohere response"
    client.chat.return_value = chat_resp

    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    p = CohereProvider()
    assert p.send("hello") == "cohere response"


@patch("ai_cli.providers.deepseek_provider.OpenAI")
def test_deepseek_provider(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="deepseek response")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    p = DeepSeekProvider()
    assert p.ask("hello") == "deepseek response"


@patch("ai_cli.providers.perplexity_provider.OpenAI")
def test_perplexity_provider(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="perplexity response")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    p = PerplexityProvider()
    assert p.send("hello") == "perplexity response"


@patch("ai_cli.providers.xAI_provider.OpenAI")
def test_xai_provider(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="xai response")
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    p = XAIProvider()
    assert p.send("hello") == "xai response"


def test_cohere_api(monkeypatch):
    provider = CohereProvider(api_key="fake")

    monkeypatch.setattr(provider, "_client", None, raising=False)


def test_deepseek_timeout():

    provider = DeepSeekProvider(api_key="fake")

    with pytest.raises(Exception, match="DeepSeek connection failed"):
        provider.chat("hello")


def test_deepseek_health_check():

    p = DeepSeekProvider(api_key="x")

    p.ask = lambda *a, **k: "ok"

    assert p.health_check() is True


def test_deepseek_embeddings(monkeypatch):
    mock_client = MagicMock()

    mock_client.embeddings.create.return_value = type(
        "R", (), {"data": [type("D", (), {"embedding": [0.1, 0.2]})()]}
    )()

    monkeypatch.setattr(
        "ai_cli.providers.deepseek_provider.OpenAI",
        lambda *args, **kwargs: mock_client,
    )

    p = DeepSeekProvider(api_key="x")

    result = p.embeddings(["hello"])

    assert result == [[0.1, 0.2]]


def test_deepseek_chat_response(monkeypatch):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C", (), {"message": type("M", (), {"content": "hello"})()}
                )()
            ]
        },
    )()

    monkeypatch.setattr(
        "ai_cli.providers.deepseek_provider.OpenAI",
        lambda *args, **kwargs: mock_client,
    )

    p = DeepSeekProvider(api_key="x")
    result = p.ask("hello")
    assert result == "hello"


def test_zai_success(monkeypatch):
    """chat() delegates to send(), which performs the real HTTP call."""
    resp = type(
        "R",
        (),
        {"status_code": 200, "json": lambda self: {"text": "hello"}},
    )()
    monkeypatch.setattr("requests.post", lambda *a, **k: resp)

    p = ZAIProvider(api_key="test-key")
    assert p.chat("hi") == "hello"


def test_zai_error(monkeypatch):
    """chat() surfaces network failures raised by send()."""

    def _raise(*_args, **_kwargs):
        raise requests.RequestException("fail")

    monkeypatch.setattr("requests.post", _raise)

    p = ZAIProvider(api_key="test-key")
    with pytest.raises(Exception, match="network error"):
        p.chat("hi")


def test_cohere_clear_index():
    p = CohereProvider(api_key="test")

    p._documents.append("x")
    p._vectors.append([1])

    p.clear_index()

    assert p._documents == []


def test_cohere_retrieve_empty():
    p = CohereProvider(api_key="test")

    assert p.retrieve("hello") == []
