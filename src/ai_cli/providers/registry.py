from __future__ import annotations

import os
from typing import Any, Type
from importlib import import_module

from ai_cli.providers.factory import build_provider as _build_provider

from ai_cli.providers.spec import ProviderRequest

_PROVIDER_SPECS = {
    "openai": (
        "ai_cli.providers.openai_provider",
        "OpenAIProvider",
    ),
    "gemini": (
        "ai_cli.providers.gemini_provider",
        "GeminiProvider",
    ),
    "cohere": (
        "ai_cli.providers.cohere_provider",
        "CohereProvider",
    ),
    "deepseek": (
        "ai_cli.providers.deepseek_provider",
        "DeepSeekProvider",
    ),
    "perplexity": (
        "ai_cli.providers.perplexity_provider",
        "PerplexityProvider",
    ),
    "xai": (
        "ai_cli.providers.xAI_provider",
        "XAIProvider",
    ),
    "zai": (
        "ai_cli.providers.zAI_provider",
        "ZAIProvider",
    ),
}

_INITIALIZED = False

def _ensure_initialized():
    global _INITIALIZED
    if _INITIALIZED:
        return
    from ai_cli.providers.bootstrap import init_providers
    init_providers()
    _INITIALIZED = True

DEFAULT_PROVIDER = "openai"

def build_provider(name: str, **kwargs):
    request = ProviderRequest(
                              provider=name,
                              model=kwargs.get("model"),
                              api_key=kwargs.get("api_key"),
                              kwargs=kwargs,
                             )
    return _build_provider(request)



PROVIDER_MAP: dict[str, Type] = {}
CHAT_PROVIDERS: dict[str, Type] = {}
EMBEDDING_PROVIDERS: dict[str, Type] = {}

def register_provider(name: str, cls: Type) -> None:
    PROVIDER_MAP[name] = cls

def register_chat_provider(name: str, cls: Type) -> None:

    CHAT_PROVIDERS[name] = cls
    PROVIDER_MAP[name] = cls

def register_embedding_provider(name: str, cls: Type) -> None:
    EMBEDDING_PROVIDERS[name] = cls
    PROVIDER_MAP[name] = cls

def get_chat_provider(name: str, **kwargs):
    if not CHAT_PROVIDERS:
        from ai_cli.providers.bootstrap import init_providers
        init_providers()
    cls = CHAT_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown chat provider: {name}")
    return cls(**kwargs)

def get_embedding_provider(name: str, **kwargs):
    if not EMBEDDING_PROVIDERS:
        from ai_cli.providers.bootstrap import init_providers
        init_providers()
    cls = EMBEDDING_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown embedding provider: {name}")
    return cls(**kwargs)

def list_providers():
    return sorted(PROVIDER_MAP.keys())

def load_plugins():
    """
    Compatibility shim.
    Safe no-op.
    """
    return None
