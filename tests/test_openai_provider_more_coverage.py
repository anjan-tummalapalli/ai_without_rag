from unittest.mock import MagicMock

import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.openai_provider import OpenAIProvider


def test_openai_provider_raises_when_package_missing(monkeypatch):
    monkeypatch.setattr(
        "ai_cli.providers.openai_provider.OPENAI_AVAILABLE", False
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


def test_openai_provider_raises_on_empty_content(monkeypatch):
    p = OpenAIProvider(api_key="x")

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=None))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(p, "client", client, raising=False)

    with pytest.raises(ProviderRequestError, match="empty content"):
        p.send("hello")


def test_openai_provider_returns_stripped_content(monkeypatch):
    p = OpenAIProvider(api_key="x")

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="  hi there  "))]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(p, "client", client, raising=False)

    assert p.send("hello") == "hi there"


def test_openai_provider_is_ready_true_when_key_set():
    p = OpenAIProvider(api_key="x")
    assert p.is_ready() is True


def test_openai_provider_is_ready_false_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = OpenAIProvider(api_key="x")
    p.api_key = None
    assert p.is_ready() is False