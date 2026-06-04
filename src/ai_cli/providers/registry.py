from typing import Type, Any, Dict
from ai_cli.providers.base import EchoProvider, AIProvider, ProviderMetadata
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider

PROVIDERS: Dict[str, Any] = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "cohere": CohereProvider,
    "deepseek": DeepSeekProvider,
    "perplexity": PerplexityProvider,
    "xai": XAIProvider,
    "zai": ZAIProvider,
}

# Backward compatibility alias

PROVIDER_MAP = PROVIDERS

def register_provider(name: str, provider_cls: Type[AIProvider], metadata: ProviderMetadata) -> None:
    """Register a provider class under a name and ensure metadata exists."""
    PROVIDER_MAP[name.lower()] = provider_cls
    PROVIDERS.setdefault(name.lower(), metadata)


def build_provider(*, provider_name: str, **kwargs):
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