from __future__ import annotations

import os
import sys
import types
import contextlib
from unittest.mock import patch, MagicMock

# Lightweight pytest fallback (keeps original behavior if pytest not present)
try:
    import pytest
except Exception:
    import re
    import contextlib as _contextlib

    class _RaisesContext:
        def __init__(self, expected_exception, match=None):
            self.expected_exception = expected_exception
            self.match = match
            self.exception = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise AssertionError(f"{self.expected_exception} was not raised")
            if not issubclass(exc_type, self.expected_exception):
                return False
            if self.match is not None and not re.search(self.match, str(exc)):
                raise AssertionError(
                    f"exception message {exc!r} does not match {self.match!r}"
                )
            self.exception = exc
            return True

    class _PyTestStub(types.ModuleType):
        def raises(self, expected_exception, match=None):
            return _RaisesContext(expected_exception, match=match)

    pytest = _PyTestStub("pytest")
    sys.modules["pytest"] = pytest

# Minimal requests fallback so tests run even without requests installed
try:
    import requests
except Exception:
    requests = types.ModuleType("requests")

    class RequestException(Exception):
        pass

    def _post(*args, **kwargs):
        raise RequestException("requests is not installed")

    requests.RequestException = RequestException
    requests.post = _post
    sys.modules["requests"] = requests

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.zAI_provider import ZAIProvider


# Helpers to reduce repetition in tests
@contextlib.contextmanager
def _provider_env(api_key: str | None = "test-key", base: str | None = None, clear: bool = False):
    env = {}
    if api_key is not None:
        env["ZAI_API_KEY"] = api_key
    if base is not None:
        env["ZAI_API_BASE"] = base
    with patch.dict(os.environ, env, clear=clear):
        yield ZAIProvider()


def _make_resp(status: int = 200, json_data: dict | None = None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


def _assert_post_called(mock_post, api_key: str, model: str, prompt: str):
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {api_key}"
    assert kwargs["json"] == {"model": model, "prompt": prompt}


def test_zai_provider_init():
    with _provider_env(api_key="test-key", base="https://custom.z.ai/v1"):
        provider = ZAIProvider()
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.z.ai/v1"
        assert provider.model == "zai-small"
        assert provider.provider_name == "z.ai"


def test_zai_provider_missing_key():
    # No env and provider.api_key empty -> raises
    with _provider_env(api_key=None, clear=True) as provider:
        provider.api_key = ""  # ensure empty
        with pytest.raises(ProviderRequestError, match="z.AI API key not configured"):
            provider.send("Hello")


@patch("requests.post")
def test_zai_provider_send_success_text(mock_post):
    mock_post.return_value = _make_resp(json_data={"text": "hello from zAI"})
    with _provider_env(api_key="test-key") as provider:
        res = provider.send("Hello")
        assert res == "hello from zAI"
        _assert_post_called(mock_post, "test-key", provider.model, "Hello")


@patch("requests.post")
def test_zai_provider_send_success_choices(mock_post):
    mock_post.return_value = _make_resp(
        json_data={"choices": [{"message": {"content": "hello choices"}}]}
    )
    with _provider_env(api_key="test-key") as provider:
        res = provider.send("Hello")
        assert res == "hello choices"
        _assert_post_called(mock_post, "test-key", provider.model, "Hello")


@patch("requests.post")
def test_zai_provider_network_error(mock_post):
    mock_post.side_effect = requests.RequestException("connection failed")
    with _provider_env(api_key="test-key") as provider:
        with pytest.raises(ProviderRequestError, match="network error"):
            provider.send("Hello")


@patch("requests.post")
def test_zai_provider_http_error(mock_post):
    mock_post.return_value = _make_resp(status=500, json_data={"error": "internal error"})
    with _provider_env(api_key="test-key") as provider:
        with pytest.raises(ProviderRequestError, match="z.AI error 500"):
            provider.send("Hello")
