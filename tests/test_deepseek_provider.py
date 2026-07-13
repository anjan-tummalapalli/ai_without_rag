import pytest

from ai_cli.providers.deepseek_provider import DeepSeekProvider


def test_deepseek_send_success(monkeypatch):
    class FakeChat:
        def create(self, **kwargs):
            return type("Resp", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]})()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeChat()})()

    provider = DeepSeekProvider(api_key="x")
    provider.client = FakeClient()
    assert "ok" in provider.send("hello")

def test_deepseek_missing_client_raises():
    provider = DeepSeekProvider(api_key=None)
    with pytest.raises(RuntimeError, match="DeepSeek request failed"):
        provider.send("hello")

