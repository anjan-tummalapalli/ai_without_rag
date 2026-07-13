import pytest
from ai_cli.providers.deepseek_provider import DeepSeekProvider


def test_deepseek_missing_key():
    with pytest.raises(Exception):
        DeepSeekProvider(api_key=None).send("hello")


def test_deepseek_success(monkeypatch):
    class FakeResp:
        choices = [type("C", (), {"message": type("M", (), {"content": "ok"})()})]

    class FakeCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    p = DeepSeekProvider(api_key="x")
    p.client = FakeClient()
    assert "ok" in p.send("hello")
