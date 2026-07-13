from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.providers import factory
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.config import resolve_api_key
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.openai import OpenAIProvider
from ai_cli.providers.spec import ProviderRequest
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider


class FakeClient:
    def __init__(self, response):
        self.response = response

    def send(self, *args, **kwargs):
        return self.response


def test_gemini_provider_basic():
    provider = GeminiProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None


def test_cohere_provider_basic():
    provider = CohereProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None


def test_xai_provider():
    provider = XAIProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None


def test_zai_provider():
    provider = ZAIProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None


def test_deepseek_provider():
    with patch("ai_cli.providers.deepseek_provider.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        provider = DeepSeekProvider(api_key="test")
        mock_choice = MagicMock()
        mock_choice.message.content = "hello back"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        result = provider.send("hello")

    assert result == "hello back"


def test_minimal_openai_provider():
    provider = OpenAIProvider(
        api_key="dummy-key",
        model="dummy-model",
    )

    assert provider.api_key == "dummy-key"
    assert provider.model == "dummy-model"

    response = provider.send("Hello")

    assert response == "OpenAI response: Hello"


def test_build_provider_success(monkeypatch):
    class DummyProvider:
        def __init__(self, provider_name, model, **kwargs):
            self.provider_name = provider_name
            self.model = model
            self.kwargs = kwargs

    monkeypatch.setattr(
        factory,
        "_PROVIDERS",
        {"openai": DummyProvider},
    )

    monkeypatch.setenv("OPENAI_API_KEY", "env-secret")

    request = SimpleNamespace(
        provider="auto",
        api_key=None,
        model="gpt-test",
        kwargs={"temperature": 0.2},
    )

    provider = factory.build_provider(request)

    assert isinstance(provider, DummyProvider)
    assert provider.provider_name == "openai"
    assert provider.model == "gpt-test"
    assert provider.kwargs["api_key"] == "env-secret"
    assert provider.kwargs["temperature"] == 0.2


def test_build_provider_unknown(monkeypatch):
    monkeypatch.setattr(factory, "_PROVIDERS", {})

    request = SimpleNamespace(
        provider="missing",
        api_key=None,
        model=None,
        kwargs={},
    )

    with pytest.raises(ValueError, match="Unknown provider"):
        factory.build_provider(request)


def test_provider_request_defaults():
    req = ProviderRequest(provider="openai")

    assert req.provider == "openai"
    assert req.model is None
    assert req.api_key is None
    assert req.kwargs is None


def test_provider_request_all_fields():
    req = ProviderRequest(
        provider="gemini",
        model="gemini-2.5-pro",
        api_key="secret",
        kwargs={"temperature": 0.7},
    )

    assert req.provider == "gemini"
    assert req.model == "gemini-2.5-pro"
    assert req.api_key == "secret"
    assert req.kwargs == {"temperature": 0.7}


def test_resolve_api_key_explicit():
    assert resolve_api_key("openai", "explicit-key") == "explicit-key"
