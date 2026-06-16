# Provider registry for ai_cli.
"""
This module maintains mappings of provider names to their implementing classes.
It avoids eager imports to prevent circular import issues. Provider modules
register themselves via ``register_provider`` (and related functions) when they
are imported. ``ai_cli.providers.bootstrap.init_providers`` loads all provider
modules lazily.
"""
from ai_cli.providers.base import BaseProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.rag.in_memory import InMemoryVectorStore

# Core mappings populated by provider modules at import time
PROVIDER_MAP: dict[str, type] = {}

# Legacy name used by plugins/tests
PROVIDERS = PROVIDER_MAP
CHAT_PROVIDERS: dict[str, type] = {}
EMBEDDING_PROVIDERS: dict[str, type] = {}

# Backward compatibility alias
PROVIDERS = PROVIDER_MAP
DEFAULT_PROVIDER = "openai"

class ProviderRegistry(dict):
    def __getitem__(self, key):
        return PROVIDER_MAP.get(key)


PROVIDERS = ProviderRegistry()

def register_provider(name: str, cls: type, metadata=None) -> None:
    """
    Register a provider.

    Args:
        name: Provider name.
        cls: Provider class.
        metadata: Optional provider metadata.
    """
    PROVIDER_MAP[name] = cls

    def decorator(provider_cls):
        PROVIDER_MAP[name] = provider_cls
        return provider_cls

    if cls is None:
        return decorator

    PROVIDER_MAP[name] = cls
    return cls


def register_chat_provider(name: str, cls: type) -> None:
    """Register a chat‑capable provider.

    This also registers the class as a generic provider so that ``build_provider``
    can retrieve it.
    """
    CHAT_PROVIDERS[name] = cls
    PROVIDER_MAP[name] = cls


def register_embedding_provider(name: str, cls: type) -> None:
    """Register an embedding provider.

    Embedding providers are also usable as generic providers.
    """
    EMBEDDING_PROVIDERS[name] = cls
    PROVIDER_MAP[name] = cls


def get_chat_provider(name: str, **kwargs):
    """Retrieve an instantiated chat provider.

    The provider class must have been registered via ``register_chat_provider``.
    """
    ensure_initialized()
    if name not in CHAT_PROVIDERS:
        raise ValueError(f"Unknown chat provider: {name}")
    return CHAT_PROVIDERS[name](**kwargs)


def get_embedding_provider(name: str, **kwargs):
    """Retrieve an instantiated embedding provider.
    """
    ensure_initialized()
    if name not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {name}")
    return EMBEDDING_PROVIDERS[name](**kwargs)


def list_providers():
    """Return a sorted list of all registered provider names."""
    ensure_initialized()
    return sorted(PROVIDER_MAP.keys())


def ensure_initialized():
    """Ensure all providers are loaded.

    Calls init_providers unconditionally to guarantee that the registry contains
    every provider, including "echo" which may have been missing in earlier
    initializations. Re‑registering is safe because the mappings are overwritten
    with the same class objects.
    """
    from ai_cli.providers.bootstrap import init_providers
    init_providers()


def build_provider(name: str, **kwargs):
    """Factory for a generic provider.

    ``ensure_initialized`` should be called beforehand (via ``ensure_initialized``
    or ``ensure_initialized`` in callers) to populate the registry.
    """
    ensure_initialized()
    if name not in PROVIDER_MAP:
        raise ValueError(f"Unknown provider: {name}")
    return PROVIDER_MAP[name](**kwargs)

provider_all = {
    "BaseProvider": BaseProvider,
    "CohereProvider": CohereProvider,
    "XAIProvider": XAIProvider,
    "InMemoryVectorStore": InMemoryVectorStore,
}
