from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import pytest
import requests

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.zAI_provider import ZAIProvider


def test_zai_provider_init():
    # Test initialization with default meta
    with patch.dict(os.environ, {"ZAI_API_KEY": "test-key", "ZAI_API_BASE": "https://custom.z.ai/v1"}):
        provider = ZAIProvider()
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.z.ai/v1"
        assert provider.model == "zai-small"
        assert provider.provider_name == "z.ai"


def test_zai_provider_missing_key():
    # Test error when key is missing
    with patch.dict(os.environ, {}, clear=True):
        provider = ZAIProvider()
        provider.api_key = ""  # ensure empty
        with pytest.raises(ProviderRequestError, match="z.AI API key not configured"):
            provider.send("Hello")


@patch("requests.post")
def test_zai_provider_send_success_text(mock_post):
    # Test successful text response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"text": "hello from zAI"}
    mock_post.return_value = mock_resp

    with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}):
        provider = ZAIProvider()
        res = provider.send("Hello")
        assert res == "hello from zAI"
        
        # Verify request parameters
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert kwargs["json"] == {"model": "zai-small", "prompt": "Hello"}


@patch("requests.post")
def test_zai_provider_send_success_choices(mock_post):
    # Test successful choices response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "hello choices"
                }
            }
        ]
    }
    mock_post.return_value = mock_resp

    with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}):
        provider = ZAIProvider()
        res = provider.send("Hello")
        assert res == "hello choices"


@patch("requests.post")
def test_zai_provider_network_error(mock_post):
    # Test network exception
    mock_post.side_effect = requests.RequestException("connection failed")

    with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}):
        provider = ZAIProvider()
        with pytest.raises(ProviderRequestError, match="network error"):
            provider.send("Hello")


@patch("requests.post")
def test_zai_provider_http_error(mock_post):
    # Test status >= 400
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {"error": "internal error"}
    mock_post.return_value = mock_resp

    with patch.dict(os.environ, {"ZAI_API_KEY": "test-key"}):
        provider = ZAIProvider()
        with pytest.raises(ProviderRequestError, match="z.AI error 500"):
            provider.send("Hello")
