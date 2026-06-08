from typing import Type, Dict
from ai_cli.providers.base import EchoProvider, AIProvider, ProviderMetadata
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider

# Map provider name -> provider class
PROVIDERS: Dict[str, Type[AIProvider]] = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "cohere": CohereProvider,
    "deepseek": DeepSeekProvider,
    "perplexity": PerplexityProvider,
    "xai": XAIProvider,
    "zai": ZAIProvider,
}

# Separate metadata map to avoid storing metadata in the provider-class map
PROVIDER_METADATA: Dict[str, ProviderMetadata] = {}

# Backward compatibility alias
PROVIDER_MAP = PROVIDERS

def register_provider(name: str, provider_cls: Type[AIProvider], metadata: ProviderMetadata) -> None:
    """Register a provider class and its metadata under a normalized name."""
    key = name.lower()
    PROVIDERS[key] = provider_cls
    PROVIDER_METADATA[key] = metadata


def load_plugins() -> None:
    """
    Placeholder for plugin loading to preserve compatibility with imports.
    """
    return None

def build_provider(provider_name: str, **kwargs):
    cls = PROVIDER_MAP.get(provider_name)
    if not cls:
        raise ValueError(f"Unknown provider: {provider_name}")
    return cls(**kwargs)

def get_chat_provider(provider_name: str = "auto", **kwargs):
    if provider_name == "auto":
        return AutoProvider()
    return build_provider(provider_name, **kwargs)