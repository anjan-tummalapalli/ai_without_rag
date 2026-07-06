from types import SimpleNamespace

import pytest

import ai_cli.providers.factory as factory


class DummyProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_factory_build_provider_success(monkeypatch):
    monkeypatch.setattr(
        factory,
        "_PROVIDERS",
        {"openai": DummyProvider},
    )

    monkeypatch.setattr(
        factory,
        "resolve_provider_name",
        lambda p: "openai",
    )

    monkeypatch.setattr(
        factory,
        "resolve_api_key",
        lambda provider, key: "resolved-key",
    )

    req = SimpleNamespace(
        provider="auto",
        api_key=None,
        model="gpt-test",
        kwargs={"temperature": 0.7},
    )

    provider = factory.build_provider(req)

    assert isinstance(provider, DummyProvider)
    assert provider.kwargs["provider_name"] == "openai"
    assert provider.kwargs["model"] == "gpt-test"
    assert provider.kwargs["api_key"] == "resolved-key"
    assert provider.kwargs["temperature"] == 0.7

def test_no_import_time_crashes():
    pass

def test_factory_unknown_provider(monkeypatch):
    monkeypatch.setattr(factory, "_PROVIDERS", {})

    monkeypatch.setattr(
        factory,
        "resolve_provider_name",
        lambda p: "does-not-exist",
    )

    req = SimpleNamespace(
        provider="bad",
        api_key=None,
        model=None,
        kwargs={},
    )

    with pytest.raises(ValueError, match="Unknown provider"):
        factory.build_provider(req)


def test_factory_build_provider_none_kwargs(monkeypatch):
    monkeypatch.setattr(
        factory,
        "_PROVIDERS",
        {"openai": DummyProvider},
    )

    monkeypatch.setattr(
        factory,
        "resolve_provider_name",
        lambda p: "openai",
    )

    monkeypatch.setattr(
        factory,
        "resolve_api_key",
        lambda provider, key: "abc123",
    )

    req = SimpleNamespace(
        provider="openai",
        api_key="ignored",
        model=None,
        kwargs=None,
    )

    provider = factory.build_provider(req)

    assert provider.kwargs["api_key"] == "abc123"
    assert provider.kwargs["provider_name"] == "openai"
    assert provider.kwargs["model"] is None