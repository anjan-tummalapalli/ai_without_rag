import pytest
import requests

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.contracts import ChatProvider
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

def test_send_impl_empty_response(monkeypatch):
    provider = ZAIProvider()
    provider.api_key = "key"

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            raise ValueError()

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResp(),
    )

    with pytest.raises(ProviderRequestError, match="empty response"):
        provider._send_impl("hello")

def test_send_impl_choices_without_message(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    import ai_cli.providers.zAI_provider as mod

    def fake_post(*args, **kwargs):
        return DummyResponse(
            json_data={
                "choices": [
                    {
                        "foo": "bar",
                    }
                ]
            }
        )

    monkeypatch.setattr(mod.requests, "post", fake_post)

    provider = ZAIProvider()

    assert provider._send_impl("hello") == '{"choices": [{"foo": "bar"}]}'

def test_send_impl_choices_not_dict(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    import ai_cli.providers.zAI_provider as mod

    def fake_post(*args, **kwargs):
        return DummyResponse(
            json_data={
                "choices": [
                    "plain string",
                ]
            }
        )

    monkeypatch.setattr(mod.requests, "post", fake_post)

    provider = ZAIProvider()

    assert provider._send_impl("hello") == '{"choices": ["plain string"]}'

def test_send_impl_json_dump_failure(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    import ai_cli.providers.zAI_provider as mod

    class BadObject:
        pass

    class DummyBadResponse:
        status_code = 200
        text = ""

        def json(self):
            return BadObject()

    monkeypatch.setattr(
        mod.requests,
        "post",
        lambda *a, **k: DummyBadResponse(),
    )

    provider = ZAIProvider()

    result = provider._send_impl("hello")

    assert "BadObject" in result

def test_send_impl_output_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(
        mod.requests,
        "post",
        lambda *a, **k: DummyResponse(json_data={"output": "hello"}),
    )

    provider = ZAIProvider()

    assert provider._send_impl("hi") == "hello"

def test_send_impl_result_key(monkeypatch):
    provider = ZAIProvider()
    provider.api_key = "key"

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": "success"}

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResp(),
    )

    assert provider._send_impl("hi") == "success"

def test_send_impl_non_json_text_response(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    class DummyResponse:
        status_code = 200
        text = "plain text response"

        def json(self):
            raise ValueError("not json")

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: DummyResponse())

    provider = ZAIProvider()

    assert provider._send_impl("hello") == "plain text response"

def test_send_impl_empty_text(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    class DummyResponse:
        status_code = 200
        text = ""

        def json(self):
            raise ValueError()

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: DummyResponse())

    provider = ZAIProvider()

    with pytest.raises(
        ProviderRequestError,
        match="empty response",
    ):
        provider._send_impl("hello")

def test_send_impl_choice_text(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    class DummyResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "choices": [
                    {
                        "text": "answer",
                    }
                ]
            }

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: DummyResponse())

    provider = ZAIProvider()

    assert provider._send_impl("hello") == "answer"

def test_send_impl_json_dump(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "abc")

    class DummyResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"foo": ["bar"]}

    import ai_cli.providers.zAI_provider as mod

    monkeypatch.setattr(mod.requests, "post", lambda *a, **k: DummyResponse())

    provider = ZAIProvider()

    assert provider._send_impl("hello") == '{"foo": ["bar"]}'

def test_chat_provider_methods():
    class Dummy(ChatProvider):
        def ask(self, prompt, **kwargs):
            return prompt

    assert Dummy().ask("hi") == "hi"

def test_send_impl_choices_message_without_content(monkeypatch):
    provider = ZAIProvider()
    provider.api_key = "key"

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {}
                    }
                ]
            }

    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: FakeResp(),
    )

    result = provider._send_impl("hello")

    assert "choices" in result
