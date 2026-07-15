import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ai_cli.plugins.builtins as builtins
from ai_cli.core.exceptions import (
    ProviderConfigurationError,
    ProviderRequestError,
    ResponseValidationError,
)

# ---------------------------------------------------------------------
# OpenAI Provider
# ---------------------------------------------------------------------


def test_openai_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = builtins.OpenAIProvider()

    with pytest.raises(ProviderConfigurationError):
        provider.send("hello")


def test_openai_import_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )
    provider = builtins.OpenAIProvider()
    with pytest.raises(ProviderConfigurationError):
        provider.send("hello")


def test_openai_request_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("boom")

    fake_sdk = SimpleNamespace(
        OpenAI=lambda **kwargs: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
        raising=False,
    )
    provider = builtins.OpenAIProvider()

    with pytest.raises(ProviderRequestError):
        provider.send("hello")


def test_openai_invalid_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    response = SimpleNamespace(choices=[])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = response

    fake_sdk = SimpleNamespace(
        OpenAI=lambda **kwargs: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
        raising=False,
    )

    provider = builtins.OpenAIProvider()

    with pytest.raises(ResponseValidationError):
        provider.send("hello")


def test_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    message = SimpleNamespace(content=" hello ")
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = response

    fake_sdk = SimpleNamespace(
        OpenAI=lambda **kwargs: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
        raising=False,
    )

    provider = builtins.OpenAIProvider()

    assert provider.send("hello") == "hello"


# ---------------------------------------------------------------------
# OpenAI Compatible Providers
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls,env_name",
    [
        (builtins.PerplexityProvider, "PERPLEXITY_API_KEY"),
        (builtins.DeepSeekProvider, "DEEPSEEK_API_KEY"),
        (builtins.GroqProvider, "GROQ_API_KEY"),
        (builtins.OpenRouterProvider, "OPENROUTER_API_KEY"),
        (builtins.TogetherProvider, "TOGETHER_API_KEY"),
        (builtins.FireworksProvider, "FIREWORKS_API_KEY"),
        (builtins.XAIProvider, "XAI_API_KEY"),
    ],
)
def test_openai_compatible_missing_key(monkeypatch, provider_cls, env_name):
    monkeypatch.delenv(env_name, raising=False)

    provider = provider_cls()

    with pytest.raises(ProviderConfigurationError):
        provider.send("hello")


@pytest.mark.parametrize(
    "provider_cls,env_name",
    [
        (builtins.PerplexityProvider, "PERPLEXITY_API_KEY"),
        (builtins.DeepSeekProvider, "DEEPSEEK_API_KEY"),
        (builtins.GroqProvider, "GROQ_API_KEY"),
        (builtins.OpenRouterProvider, "OPENROUTER_API_KEY"),
        (builtins.TogetherProvider, "TOGETHER_API_KEY"),
        (builtins.FireworksProvider, "FIREWORKS_API_KEY"),
        (builtins.XAIProvider, "XAI_API_KEY"),
    ],
)
def test_openai_compatible_success(monkeypatch, provider_cls, env_name):
    monkeypatch.setenv(env_name, "dummy")

    message = SimpleNamespace(content="success")
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = response

    fake_sdk = SimpleNamespace(
        OpenAI=lambda **kwargs: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
        raising=False,
    )

    provider = provider_cls()

    assert provider.send("hello") == "success"


# ---------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------


def test_gemini_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = builtins.GeminiProvider()

    with pytest.raises(ProviderConfigurationError):
        provider.send("hello")


# ---------------------------------------------------------------------
# Cohere
# ---------------------------------------------------------------------


def test_cohere_missing_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    provider = builtins.CohereProvider()

    with pytest.raises(ProviderConfigurationError):
        provider.send("hello")


def test_openai_empty_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = response

    fake_sdk = SimpleNamespace(OpenAI=lambda **kwargs: fake_client)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.OpenAIProvider()

    with pytest.raises(ResponseValidationError):
        provider.send("hello")


def test_openai_compatible_import_failure(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "dummy")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )

    provider = builtins.PerplexityProvider()

    with pytest.raises(ProviderConfigurationError):
        provider.send("hello")


def test_openai_compatible_request_failure(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "dummy")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("boom")

    fake_sdk = SimpleNamespace(OpenAI=lambda **kwargs: fake_client)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.PerplexityProvider()

    with pytest.raises(ProviderRequestError):
        provider.send("hello")


def test_openai_compatible_invalid_response(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "dummy")

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[]
    )

    fake_sdk = SimpleNamespace(OpenAI=lambda **kwargs: fake_client)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.PerplexityProvider()

    with pytest.raises(ResponseValidationError):
        provider.send("hello")


def test_openai_compatible_empty_response(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "dummy")

    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None))]
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = response

    fake_sdk = SimpleNamespace(OpenAI=lambda **kwargs: fake_client)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.PerplexityProvider()

    with pytest.raises(ResponseValidationError):
        provider.send("hello")


def test_gemini_provider_init():
    provider = builtins.GeminiProvider()
    assert provider.provider_name == "gemini"


def test_cohere_import_failure(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError()),
    )

    provider = builtins.CohereProvider()

    with pytest.raises(ProviderConfigurationError):
        provider.send("hello")


def test_cohere_request_failure(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")

    fake_client = MagicMock()
    fake_client.generate.side_effect = RuntimeError("boom")

    fake_sdk = SimpleNamespace(
        Client=lambda key: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.CohereProvider()

    with pytest.raises(ProviderRequestError):
        provider.send("hello")


def test_cohere_invalid_response(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")

    fake_client = MagicMock()
    fake_client.generate.return_value = SimpleNamespace(generations=[])

    fake_sdk = SimpleNamespace(
        Client=lambda key: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.CohereProvider()

    with pytest.raises(ResponseValidationError):
        provider.send("hello")


def test_cohere_success(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")

    generation = SimpleNamespace(text=" hello ")

    fake_client = MagicMock()
    fake_client.generate.return_value = SimpleNamespace(
        generations=[generation]
    )

    fake_sdk = SimpleNamespace(
        Client=lambda key: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.CohereProvider()

    assert provider.send("hi") == "hello"


def test_cohere_empty_response(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "dummy")

    generation = SimpleNamespace(text=None)

    fake_client = MagicMock()
    fake_client.generate.return_value = SimpleNamespace(
        generations=[generation]
    )

    fake_sdk = SimpleNamespace(
        Client=lambda key: fake_client,
    )

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_sdk,
    )

    provider = builtins.CohereProvider()

    with pytest.raises(ResponseValidationError):
        provider.send("hello")
