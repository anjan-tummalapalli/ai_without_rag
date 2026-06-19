from ai_cli.ai_chat import ask


def test_ai_chat_ask_executes():
    assert "mock:hello" in ask("hello")