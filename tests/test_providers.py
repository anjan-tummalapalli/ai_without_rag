from __future__ import annotations

from unittest.mock import MagicMock, patch

from ai_cli.providers.base import EchoProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider


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