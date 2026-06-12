import pytest

from ai_cli.providers.registry import (
    register_provider,
    build_provider,
    list_providers,
)

class DummyProvider:
    pass


def test_register_and_build_provider():
    register_provider(
        "dummy",
        DummyProvider
    )
    obj = build_provider("dummy")
    assert isinstance(obj, DummyProvider)


def test_unknown_provider():
    with pytest.raises(ValueError):
        build_provider("does-not-exist")


def test_list_providers():
    providers = list_providers()
    assert isinstance(providers, list)
