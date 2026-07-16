from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.auto_provider import AutoProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.registry import PROVIDER_MAP


def test_deepseek_missing_key():
    with pytest.raises(RuntimeError):
        DeepSeekProvider(api_key=None).send("hello")


def test_deepseek_success():
    class FakeResp:
        def __init__(self):
            self.choices = [
                type("C", (), {"message": type("M", (), {"content": "ok"})()})()
            ]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = FakeResp()

    p = DeepSeekProvider(api_key="x")
    p.client = fake_client

    assert p.send("hello") == "ok"


class _FallbackOkProvider:
    def send(self, prompt: str) -> str:
        _ = prompt
        return "fallback_ok"

    def ask(self, prompt: str) -> str:
        return self.send(prompt)


class _UnauthorizedThenOkProvider:
    """Stub that raises an 'unauthorized' error."""

    def send(self, prompt: str) -> str:
        _ = prompt
        raise ProviderRequestError("401 unauthorized")

    def ask(self, prompt: str) -> str:
        return self.send(prompt)


def test_send_skips_unauthorized_error_and_raises() -> None:
    PROVIDER_MAP["__auto_unauthorized__"] = _UnauthorizedThenOkProvider
    ap = AutoProvider(fallback_order=["__auto_unauthorized__"])

    with pytest.raises(ProviderRequestError, match="Auto fallback exhausted"):
        ap.send("hello")


def test_send_reports_provider_not_found() -> None:
    PROVIDER_MAP["__auto_present__"] = _FallbackOkProvider
    ap = AutoProvider(fallback_order=["__auto_present__"])
    ap.fallback_order = ["__auto_missing_now__", "__auto_present__"]

    result = ap.send("hello")
    assert result == "fallback_ok"
