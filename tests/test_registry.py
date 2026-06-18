import pytest

from ai_cli.providers.registry import PROVIDER_MAP, build_provider


def test_build_provider_invalid():
    with pytest.raises(ValueError):
        build_provider("unknown_provider")

def test_build_provider_normal(monkeypatch):
    class DummyProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
    monkeypatch.setitem(PROVIDER_MAP, "dummy", DummyProvider)
    provider = build_provider("dummy", foo="bar")
    assert isinstance(provider, DummyProvider)
    assert provider.kwargs["foo"] == "bar"