from __future__ import annotations

from ai_cli.providers.base import EchoProvider
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider

PROVIDERS = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "cohere": CohereProvider,
    "deepseek": DeepSeekProvider,
    "perplexity": PerplexityProvider,
    "xai": XAIProvider,
}

# Backward compatibility alias

PROVIDER_MAP = PROVIDERS

def build_provider(provider_name: str, **kwargs):
    """
    Create a provider instance by name.
    """
    provider_name = provider_name.lower()
    if provider_name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available providers: {', '.join(PROVIDERS.keys())}"
        )
    provider_cls = PROVIDERS[provider_name]
    return provider_cls(**kwargs)

def load_plugins() -> None:
    """
    Placeholder for plugin loading.
    Existing code imports load_plugins() from registry.py.
    Keeping this function preserves compatibility even when
    plugin discovery is not currently implemented.
    """
    return None