from unittest.mock import MagicMock

import pytest
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.openai_provider import OpenAIProvider


def test_openai_provider_raises_when_package_missing(monkeypatch):
    monkeypatch.setattr(
        "ai_cli.providers.openai_provider.OpenAI", None, raising=False
    )

    with pytest.raises(ProviderRequestError, match="not installed"):
        OpenAIProvider(api_key="real-key")


def test_openai_provider_raises_on_empty_choices(monkeypatch):
    p = OpenAIProvider(api_key="x")

    response = MagicMock()
    response.choices = []
    client = MagicMock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(p, "client", client, raising=False)

    with pytest.raises(ProviderRequestError, match="no choices"):
        p.send("hello")


def test_openai_provider_handles_dict_style_message(monkeypatch):
    p = OpenAIProvider(api_key="x")

    response = MagicMock()
    response.choices = [{"message": {"content": "dict response"}}]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(p, "client", client, raising=False)

    result = p.send("hello")

    assert result == "dict response"


def test_openai_provider_raises_on_empty_content(monkeypatch):
    p = OpenAIProvider(api_key="x")

    response = MagicMock()
    response.choices = [{"message": {"content": ""}}]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(p, "client", client, raising=False)

    with pytest.raises(ProviderRequestError, match="empty content"):
        p.send("hello")


def test_openai_provider_is_ready_reflects_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "set")
    p = OpenAIProvider(api_key="x")
    assert p.is_ready() is True

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert p.is_ready() is False
