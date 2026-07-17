from __future__ import annotations
 
from collections.abc import Callable
from typing import Any
 
from ai_cli.providers.base import BaseProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.xAI_provider import XAIProvider
 
# Core mappings populated by provider modules at import time
PROVIDER_MAP: dict[str, type[BaseProvider]] = {}
CHAT_PROVIDERS: dict[str, type[BaseProvider]] = {}
DEFAULT_PROVIDER = "openai"
 
 
class ProviderRegistry(dict[str, type[BaseProvider]]):
    """Dict-like view onto ``PROVIDER_MAP`` that always reflects live state."""
 
    def __getitem__(self, key: str) -> type[BaseProvider] | None:  # type: ignore[override]
        return PROVIDER_MAP.get(key)
 
 
PROVIDERS = ProviderRegistry()
 
 
def register_provider(
    name: str,
    cls: type[BaseProvider] | None = None,
    metadata: Any | None = None,
) -> (
    type[BaseProvider]
    | Callable[[type[BaseProvider]], type[BaseProvider]]
):
    """Register a provider."""
 
    del metadata  # metadata currently unused
 
    def decorator(
        provider_cls: type[BaseProvider],
    ) -> type[BaseProvider]:
        PROVIDER_MAP[name] = provider_cls
        return provider_cls
 
    if cls is None:
        return decorator
 
    PROVIDER_MAP[name] = cls
    return cls
 
 
def register_chat_provider(
    name: str,
    cls: type[BaseProvider],
) -> None:
    """Register a chat-capable provider."""
    CHAT_PROVIDERS[name] = cls
    PROVIDER_MAP[name] = cls
 
 
def get_chat_provider(
    name: str,
    **kwargs: Any,
) -> BaseProvider:
    """Retrieve an instantiated chat provider."""
    ensure_initialized()
 
    if name not in CHAT_PROVIDERS:
        raise ValueError(f"Unknown chat provider: {name}")
 
    return CHAT_PROVIDERS[name](**kwargs)
 
 
def list_providers() -> list[str]:
    """Return a sorted list of all registered provider names."""
    ensure_initialized()
    return sorted(PROVIDER_MAP.keys())
 
 
def ensure_initialized() -> None:
    """Ensure all providers are loaded."""
    from ai_cli.providers.bootstrap import (
        init_providers,
    )
 
    init_providers()
 
 
def build_provider(
    name: str,
    **kwargs: Any,
) -> BaseProvider:
    """Factory for a generic provider."""
    ensure_initialized()
 
    if name not in PROVIDER_MAP:
        raise ValueError(f"Unknown provider: {name}")
 
    return PROVIDER_MAP[name](**kwargs)
 
 
provider_all: dict[str, type[BaseProvider]] = {
    "BaseProvider": BaseProvider,
    "CohereProvider": CohereProvider,
    "XAIProvider": XAIProvider,
}