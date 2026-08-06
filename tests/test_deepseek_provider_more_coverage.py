from unittest.mock import MagicMock

from ai_cli.providers.deepseek_provider import DeepSeekProvider


def test_deepseek_health_check_false_without_key():
    provider = DeepSeekProvider(api_key=None)
    assert provider.health_check() is False


def test_deepseek_health_check_true_on_success(monkeypatch):
    provider = DeepSeekProvider(api_key="x")
    monkeypatch.setattr(provider, "ask", lambda *a, **k: "pong")
    assert provider.health_check() is True


def test_deepseek_health_check_false_on_exception(monkeypatch):
    provider = DeepSeekProvider(api_key="x")

    def boom(*a, **k):
        raise RuntimeError("down")

    monkeypatch.setattr(provider, "ask", boom)
    assert provider.health_check() is False


def test_deepseek_send_falls_back_to_choice_text():
    class Choice:
        message = None
        text = "text-fallback"

    class Resp:
        choices = [Choice()]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = Resp()

    provider = DeepSeekProvider(api_key="x")
    provider.client = fake_client

    assert provider.send("hello") == "text-fallback"


def test_deepseek_send_returns_string_response_when_no_choices():
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = "raw string response"

    provider = DeepSeekProvider(api_key="x")
    provider.client = fake_client

    assert provider.send("hello") == "raw string response"


def test_deepseek_send_falls_back_to_str_response():
    class Choice:
        message = type("M", (), {"content": None})()

    class Resp:
        choices = [Choice()]

        def __str__(self):
            return "stringified-response"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = Resp()

    provider = DeepSeekProvider(api_key="x")
    provider.client = fake_client

    assert provider.send("hello") == "stringified-response"


def test_deepseek_chat_falls_back_to_str_response():
    class Choice:
        message = type("M", (), {"content": None})()

    class Resp:
        choices = [Choice()]

        def __str__(self):
            return "stringified-chat-response"

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = Resp()

    provider = DeepSeekProvider(api_key="x")
    provider.client = fake_client

    assert provider.chat("hello") == "stringified-chat-response"


def test_deepseek_chat_falls_back_to_choice_text():
    class Choice:
        text = "chat-text-fallback"

    class Resp:
        choices = [Choice()]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = Resp()

    provider = DeepSeekProvider(api_key="x")
    provider.client = fake_client

    assert provider.chat("hello") == "chat-text-fallback"
