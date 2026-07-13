import pytest
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider


def test_openai_provider_raises_on_api_error(monkeypatch):
    p = OpenAIProvider(api_key="x")

    class Boom(Exception):
        pass

    def fail(*args, **kwargs):
        raise Boom("fail")

    p.client = type(
        "C",
        (),
        {"chat": type("Chat", (), {"completions": type("X", (), {"create": fail})()})()},
    )()

    with pytest.raises(Exception):
        p.send("hello")


def test_xai_missing_key():
    with pytest.raises(Exception):
        XAIProvider(api_key=None).send("hello")


def test_zai_missing_key():
    with pytest.raises(Exception):
        ZAIProvider(api_key=None).send("hello")
