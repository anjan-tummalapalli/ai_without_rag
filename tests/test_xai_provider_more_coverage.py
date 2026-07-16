from unittest.mock import MagicMock, patch

import pytest
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.xAI_provider import XAIProvider


def test_xai_provider_raises_when_package_missing(monkeypatch):
    monkeypatch.setattr(
        "ai_cli.providers.xAI_provider.OpenAI", None, raising=False
    )

    with pytest.raises(
        ProviderRequestError, match="openai package is required"
    ):
        XAIProvider(api_key="real-key")


@patch("ai_cli.providers.xAI_provider.OpenAI")
def test_xai_provider_send_raises_without_api_key(openai_mock, monkeypatch):
    openai_mock.return_value = MagicMock()
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    provider = XAIProvider(api_key=None)

    with pytest.raises(ProviderRequestError, match="API key not configured"):
        provider.send("hello")
