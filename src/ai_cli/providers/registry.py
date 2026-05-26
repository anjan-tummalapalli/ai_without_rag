from __future__ import annotations
import importlib
import pkgutil
from typing import Dict, Type

from ai_cli.providers.base import AIProvider, ProviderMetadata, EchoProvider

PROVIDERS: Dict[str, ProviderMetadata] = {
    "openai": ProviderMetadata(
        name="OpenAI",
        default_model="gpt-4o",
        supported_models=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        cost_per_1k_tokens=0.01,
        avg_latency_ms=800,
    ),
    "perplexity": ProviderMetadata(
        name="Perplexity AI",
        default_model="sonar-pro",
        supported_models=["sonar", "sonar-pro"],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.009,
        avg_latency_ms=850,
    ),
    "deepseek": ProviderMetadata(
        name="DeepSeek",
        default_model="deepseek-chat",
        supported_models=[
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.003,
        avg_latency_ms=500,
    ),
    "groq": ProviderMetadata(
        name="Groq",
        default_model="llama-3.3-70b",
        supported_models=["llama-3.3-70b", "mixtral-8x7b"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.002,
        avg_latency_ms=300,
    ),
    "openrouter": ProviderMetadata(
        name="OpenRouter",
        default_model="openai/gpt-4o",
        supported_models=[
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-pro",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=200_000,
        cost_per_1k_tokens=0.005,
        avg_latency_ms=700,
    ),
    "together": ProviderMetadata(
        name="Together AI",
        default_model="meta-llama/Llama-3-70b-chat-hf",
        supported_models=[
            "meta-llama/Llama-3-70b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.004,
        avg_latency_ms=650,
    ),
    "fireworks": ProviderMetadata(
        name="Fireworks AI",
        default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        supported_models=["accounts/fireworks/models/llama-v3p1-70b-instruct"],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.004,
        avg_latency_ms=550,
    ),
    "xai": ProviderMetadata(
        name="xAI Grok",
        default_model="grok-3",
        supported_models=["grok-3", "grok-3-mini"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        cost_per_1k_tokens=0.011,
        avg_latency_ms=750,
    ),
    "gemini": ProviderMetadata(
        name="Google Gemini",
        default_model="gemini-pro",
        supported_models=[
            "gemini-pro",
            "gemini-1.5",
            "gemini-1.0",
            "gemini-mini",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=1_000_000,
        cost_per_1k_tokens=0.012,
        avg_latency_ms=600,
    ),
    "cohere": ProviderMetadata(
        name="Cohere",
        default_model="command-xlarge-nightly",
        supported_models=[
            "command-xlarge-nightly",
            "command-nightly",
            "embed-english-v2.0",
        ],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=4_096,
        cost_per_1k_tokens=0.007,
        avg_latency_ms=400,
    ),
}

# The actual mapping is populated at runtime
PROVIDER_MAP: Dict[str, Type[AIProvider]] = {"echo": EchoProvider}


def register_provider(
    name: str, provider_class: Type[AIProvider], metadata: ProviderMetadata
) -> None:
    """Register a provider dynamically."""
    PROVIDERS[name] = metadata
    PROVIDER_MAP[name] = provider_class


def build_provider(name: str, model: str | None = None) -> AIProvider:
    """Factory that constructs a provider instance by name."""
    normalized_name = name.lower()

    # Auto logic
    if normalized_name == "auto":
        from ai_cli.providers.auto_provider import AutoProvider

        return AutoProvider(model=model)

    try:
        provider_class = PROVIDER_MAP[normalized_name]
    except KeyError as exc:
        from ai_cli.core.exceptions import ProviderConfigurationError

        raise ProviderConfigurationError(f"Unknown provider '{name}'") from exc
    return provider_class(model=model)


def load_plugins() -> None:
    """Load providers from the plugins directory."""
    import ai_cli.plugins as plugins

    for _, module_name, is_pkg in pkgutil.iter_modules(
        plugins.__path__, plugins.__name__ + "."
    ):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            import logging

            logging.getLogger("ai_gateway").warning(
                f"Failed to load plugin {module_name}: {e}"
            )