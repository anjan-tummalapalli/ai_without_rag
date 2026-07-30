"""Tests for ZAIProvider, with lightweight pytest/requests fallbacks so
the suite can still run in environments where those packages aren't
installed (e.g. a minimal CI smoke-test stage)."""

from __future__ import annotations

import contextlib
import os
import sys
import types
from unittest.mock import MagicMock, patch

# Lightweight pytest fallback (keeps original behavior if pytest not present)
try:
    import pytest  # pyrefly: ignore [missing-import]
except ImportError:
    import re

    class _RaisesContext:
        """Minimal stand-in for pytest.raises' context manager: fails the
        test if the expected exception isn't raised, re-raises anything
        of the wrong type, and optionally checks the message against a
        regex, mirroring pytest's real `match=` behavior."""

        def __init__(self, expected_exception, match=None):
            """Store the expected exception type(s) and optional regex."""
            self.expected_exception = expected_exception
            self.match = match
            self.exception = None

        def __enter__(self):
            """Return self so callers can inspect .exception afterward."""
            return self

        def __exit__(self, exc_type, exc, tb):
            """Validate the raised exception's type and message."""
            if exc_type is None:
                raise AssertionError(
                    f"{self.expected_exception} was not raised"
                )
            if not issubclass(exc_type, self.expected_exception):
                return False
            if self.match is not None and not re.search(self.match, str(exc)):
                raise AssertionError(
                    f"exception message {exc!r} does not match {self.match!r}"
                )
            self.exception = exc
            return True

    # pylint: disable-next=too-few-public-methods
    class _PyTestStub(types.ModuleType):
        """Fallback module registered as `pytest` when the real package
        isn't installed; only implements the `.raises()` this file uses."""

        def raises(self, expected_exception, match=None):
            """Return a _RaisesContext, mirroring pytest.raises()."""
            return _RaisesContext(expected_exception, match=match)

    pytest = _PyTestStub("pytest")
    sys.modules["pytest"] = pytest

# Minimal requests fallback so tests run even without requests installed
try:
    import requests
except ImportError:
    requests = types.ModuleType("requests")

    class RequestException(Exception):
        """Fallback stand-in for requests.RequestException."""

    def _post(*args, **kwargs):
        """Fallback for requests.post when requests isn't installed."""
        raise RequestException("requests is not installed")

    # pyrefly: ignore [missing-attribute]
    requests.RequestException = RequestException
    requests.post = _post  # pyrefly: ignore [missing-attribute]
    sys.modules["requests"] = requests

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.zAI_provider import ZAIProvider


# Helpers to reduce repetition in tests
@contextlib.contextmanager
def _provider_env(
    api_key: str | None = "test-key",
    base: str | None = None,
    clear: bool = False,
):
    """Patch ZAI_API_KEY/ZAI_API_BASE for the duration of the `with`
    block and yield a ZAIProvider constructed under that environment."""
    env = {}
    if api_key is not None:
        env["ZAI_API_KEY"] = api_key
    if base is not None:
        env["ZAI_API_BASE"] = base
    with patch.dict(os.environ, env, clear=clear):
        yield ZAIProvider()


def _make_resp(status: int = 200, json_data: dict | None = None):
    """Build a MagicMock standing in for a requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


def _assert_post_called(
    mock_post, api_key: str, model: str | None, prompt: str
):
    """Assert requests.post was called once with the expected auth
    header and JSON payload."""
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == f"Bearer {api_key}"
    assert kwargs["json"] == {"model": model, "prompt": prompt}


def test_zai_provider_init():
    """ZAIProvider() picks up api_key/base_url from the environment and
    has the expected default model and provider_name."""
    with _provider_env(api_key="test-key", base="https://custom.z.ai/v1"):
        provider = ZAIProvider()
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://custom.z.ai/v1"
        assert provider.model == "zai-small"
        assert provider.provider_name == "z.ai"


def test_zai_provider_missing_key():
    """send() raises when no API key is configured (env cleared and
    api_key explicitly set empty)."""
    # No env and provider.api_key empty -> raises
    with _provider_env(api_key=None, clear=True) as provider:
        provider.api_key = ""  # ensure empty
        with pytest.raises(
            ProviderRequestError, match="z.AI API key not configured"
        ):
            provider.send("Hello")


@patch("requests.post")
def test_zai_provider_send_success_text(mock_post):
    """send() extracts the reply from a {"text": ...} response shape."""
    mock_post.return_value = _make_resp(json_data={"text": "hello from zAI"})
    with _provider_env(api_key="test-key") as provider:
        res = provider.send("Hello")
        assert res == "hello from zAI"
        _assert_post_called(mock_post, "test-key", provider.model, "Hello")


@patch("requests.post")
def test_zai_provider_send_success_choices(mock_post):
    """send() extracts the reply from a {"choices": [...]} response shape."""
    mock_post.return_value = _make_resp(
        json_data={"choices": [{"message": {"content": "hello choices"}}]}
    )
    with _provider_env(api_key="test-key") as provider:
        res = provider.send("Hello")
        assert res == "hello choices"
        _assert_post_called(mock_post, "test-key", provider.model, "Hello")


@patch("requests.post")
def test_zai_provider_network_error(mock_post):
    """send() raises ProviderRequestError when the HTTP call fails."""
    mock_post.side_effect = requests.RequestException("connection failed")
    with _provider_env(api_key="test-key") as provider:
        with pytest.raises(ProviderRequestError, match="network error"):
            provider.send("Hello")


@patch("requests.post")
def test_zai_provider_http_error(mock_post):
    """send() raises ProviderRequestError on a non-200 HTTP status."""
    mock_post.return_value = _make_resp(
        status=500, json_data={"error": "internal error"}
    )
    with _provider_env(api_key="test-key") as provider:
        with pytest.raises(ProviderRequestError, match="z.AI error 500"):
            provider.send("Hello")


def test_zai_provider_constructs_with_explicit_key():
    """ZAIProvider(api_key=...) uses the explicitly passed key rather
    than requiring it to come from the environment."""
    provider = ZAIProvider(api_key="fake")

    assert provider.api_key == "fake"
