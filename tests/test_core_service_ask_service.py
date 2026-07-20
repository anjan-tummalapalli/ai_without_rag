"""Tests for ai_cli.core.service.ask_service.ask()."""

from unittest.mock import patch

from ai_cli.core.service.ask_service import ask

# ask_service now delegates to AIService which calls _core_ask internally.
# We patch at that seam so the public contract (input → output) is unchanged.
_PATCH_TARGET = "ai_cli.core.service.ai_service._core_ask"


def test_ask_uses_explicit_api_key_and_returns_chat_result():
    with patch(_PATCH_TARGET, return_value="hello there") as mock_ask:
        result = ask(
            "hi",
            provider="openai",
            model="gpt-test",
            api_key="explicit-key",
            timeout=5.0,
        )

    assert result == "hello there"
    mock_ask.assert_called_once()
    call_kwargs = mock_ask.call_args.kwargs
    assert call_kwargs["prompt"] == "hi"
    assert call_kwargs["provider"] == "openai"
    assert call_kwargs["model"] == "gpt-test"
    assert call_kwargs.get("api_key") == "explicit-key"


def test_ask_falls_back_to_env_var_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    with patch(_PATCH_TARGET, return_value="env response") as mock_ask:
        result = ask("hi", provider="openai")

    assert result == "env response"
    mock_ask.assert_called_once()
    call_kwargs = mock_ask.call_args.kwargs
    assert call_kwargs["provider"] == "openai"
    # api_key should be the env var value injected by _AskServiceAdapter
    assert call_kwargs.get("api_key") == "env-key"


def test_ask_defaults_to_auto_provider_with_no_env_key(monkeypatch):
    monkeypatch.delenv("AUTO_API_KEY", raising=False)
    with patch(_PATCH_TARGET, return_value="auto response") as mock_ask:
        result = ask("hi")

    assert result == "auto response"
    mock_ask.assert_called_once()
    call_kwargs = mock_ask.call_args.kwargs
    assert call_kwargs["provider"] == "auto"
