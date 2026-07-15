"""Tests for ai_cli.core.service.ask_service.ask()."""

from unittest.mock import MagicMock, patch

from ai_cli.core.service.ask_service import ask


def test_ask_uses_explicit_api_key_and_returns_chat_result():
    mock_provider = MagicMock()
    mock_provider.chat.return_value = "hello there"

    with patch(
        "ai_cli.core.service.ask_service.get_chat_provider",
        return_value=mock_provider,
    ) as mock_get_provider:
        result = ask(
            "hi",
            provider="openai",
            model="gpt-test",
            api_key="explicit-key",
            timeout=5.0,
        )

    assert result == "hello there"
    mock_get_provider.assert_called_once_with(
        "openai",
        model="gpt-test",
        api_key="explicit-key",
        timeout=5.0,
    )
    mock_provider.chat.assert_called_once_with("hi")


def test_ask_falls_back_to_env_var_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    mock_provider = MagicMock()
    mock_provider.chat.return_value = "env response"

    with patch(
        "ai_cli.core.service.ask_service.get_chat_provider",
        return_value=mock_provider,
    ) as mock_get_provider:
        result = ask("hi", provider="openai")

    assert result == "env response"
    mock_get_provider.assert_called_once_with(
        "openai",
        model=None,
        api_key="env-key",
        timeout=None,
    )


def test_ask_defaults_to_auto_provider_with_no_env_key(monkeypatch):
    monkeypatch.delenv("AUTO_API_KEY", raising=False)
    mock_provider = MagicMock()
    mock_provider.chat.return_value = "auto response"

    with patch(
        "ai_cli.core.service.ask_service.get_chat_provider",
        return_value=mock_provider,
    ) as mock_get_provider:
        result = ask("hi")

    assert result == "auto response"
    mock_get_provider.assert_called_once_with(
        "auto",
        model=None,
        api_key=None,
        timeout=None,
    )
