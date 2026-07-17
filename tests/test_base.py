import pytest

from ai_cli.providers.base import (
    AIProvider,
    BaseProvider,
    EchoProvider,
    ProviderMetadata,
)


def test_provider_metadata_defaults_and_values():
    meta = ProviderMetadata(name="acme")
    assert meta.name == "acme"
    assert meta.supports_rag is False

    meta2 = ProviderMetadata(name="other", supports_rag=True)
    assert meta2.name == "other"
    assert meta2.supports_rag is True

def test_baseprovider_init_sets_attrs():
    bp = BaseProvider(api_key="secret", model="gpt-test")
    assert bp.api_key == "secret"
    assert bp.model == "gpt-test"

def test_baseprovider_send_not_implemented():
    bp = BaseProvider()
    with pytest.raises(NotImplementedError):
        bp.send("hello")

def test_baseprovider_ask_delegates_to_send(monkeypatch):
    bp = BaseProvider()
    called = {}
    def fake_send(prompt, **kwargs):
        called['prompt'] = prompt
        called['kwargs'] = kwargs
        return "sent:" + prompt
    monkeypatch.setattr(bp, "send", fake_send)
    out = bp.ask("hey", foo=1)
    assert out == "sent:hey"
    assert called['prompt'] == "hey"
    assert called['kwargs'] == {"foo": 1}

def test_baseprovider_chat_delegates_to_ask(monkeypatch):
    bp = BaseProvider()
    called = {}
    def fake_ask(prompt, **kwargs):
        called['prompt'] = prompt
        called['kwargs'] = kwargs
        return "asked:" + prompt
    monkeypatch.setattr(bp, "ask", fake_ask)
    out = bp.chat("chat me", bar=2)
    assert out == "asked:chat me"
    assert called['prompt'] == "chat me"
    assert called['kwargs'] == {"bar": 2}

def test_ai_provider_alias_is_baseprovider():
    assert AIProvider is BaseProvider

def test_echoprovider_basic_behavior():
    e = EchoProvider(api_key="s")
    # EchoProvider sets model to "echo" in its constructor via super()
    assert e.model == "echo"
    assert e.api_key == "s"
    # send must return the prompt prefixed
    assert e.send("hello") == "(echo) hello"
    # ask and chat should delegate to send through BaseProvider impl
    assert e.ask("ask") == "(echo) ask"
    assert e.chat("chat") == "(echo) chat"

def test_echoprovider_provider_name_present():
    assert EchoProvider.provider_name == "echo"
