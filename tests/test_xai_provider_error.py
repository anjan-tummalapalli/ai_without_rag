import pytest
from unittest.mock import patch, MagicMock
from ai_cli.providers.xAI_provider import XAIProvider

@patch("ai_cli.providers.xAI_provider.OpenAI")
def test_xai_provider_empty_response(openai_mock, monkeypatch):
    # Setup mock client that returns empty choices
    client = MagicMock()
    openai_mock.return_value = client
    resp = MagicMock()
    resp.choices = []  # No choices
    client.chat.completions.create.return_value = resp
    # Ensure env var for API key
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    provider = XAIProvider()
    # send should catch the ProviderRequestError and return placeholder
    result = provider.send("test")
    assert result == "[Error: unable to get response]"
