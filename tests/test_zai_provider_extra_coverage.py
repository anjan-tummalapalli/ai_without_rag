import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.zAI_provider import ZAIProvider


class DummyResponse:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code = status
        self._json = json_data
        self.text = text

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


def test_send_output_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    def fake_post(*args, **kwargs):
        return DummyResponse(json_data={"text": "hello"})

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", fake_post)

    p = ZAIProvider()
    assert p.send("hi") == "hello"


def test_send_choices_path(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    def fake_post(*args, **kwargs):
        return DummyResponse(
            json_data={"choices": [{"message": {"content": "choice response"}}]}
        )

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", fake_post)

    p = ZAIProvider()
    assert p.send("hi") == "choice response"


def test_send_unknown_response(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    def fake_post(*args, **kwargs):
        return DummyResponse(json_data={"foo": "bar"})

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", fake_post)

    p = ZAIProvider()

    with pytest.raises(
        ProviderRequestError,
        match="unable to coerce",
    ):
        p.send("hello")


def test_send_http_error(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    def fake_post(*args, **kwargs):
        return DummyResponse(status=500, json_data={})

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", fake_post)

    p = ZAIProvider()

    with pytest.raises(
        ProviderRequestError,
        match="z.AI error 500",
    ):
        p.send("hello")


def test_send_network_error(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    import ai_cli.providers.zAI_provider as mod

    def fake_post(*args, **kwargs):
        raise mod.requests.RequestException("boom")

    monkeypatch.setattr(mod.requests, "post", fake_post)

    p = ZAIProvider()

    with pytest.raises(
        ProviderRequestError,
        match="network error",
    ):
        p.send("hello")


def test_send_mock_api_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test")

    p = ZAIProvider()

    assert p.send("hello") == "mock:hello"


def test_send_missing_key(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    p = ZAIProvider()

    with pytest.raises(
        ProviderRequestError,
        match="API key not configured",
    ):
        p.send("hello")


def test_is_ready_true(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    assert ZAIProvider().is_ready() is True


def test_is_ready_false(monkeypatch):
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    assert ZAIProvider().is_ready() is False
