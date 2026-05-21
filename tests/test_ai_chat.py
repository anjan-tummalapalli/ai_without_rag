import pytest
from src.ai_cli.ai_chat import ask, AVAILABLE_MODELS, PROVIDERS

def test_ask_valid_provider():
    response = ask("openai", "What is the capital of France?")
    assert response is not None
    assert "Paris" in response

def test_ask_invalid_provider():
    response = ask("invalid_provider", "What is the capital of France?")
    assert response.startswith("[ERROR]")

def test_available_models():
    assert isinstance(AVAILABLE_MODELS, dict)
    assert "openai" in AVAILABLE_MODELS
    assert len(AVAILABLE_MODELS["openai"]) > 0

def test_providers():
    assert isinstance(PROVIDERS, dict)
    assert len(PROVIDERS) > 0
    assert "openai" in PROVIDERS
    assert "gemini" in PROVIDERS

def test_ask_with_model():
    response = ask("openai", "What is the capital of France?", model="gpt-4o")
    assert response is not None
    assert "Paris" in response

def test_ask_empty_prompt():
    response = ask("openai", "")
    assert response.startswith("[ERROR]")