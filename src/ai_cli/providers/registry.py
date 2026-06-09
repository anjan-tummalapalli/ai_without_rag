from __future__ import annotations
from typing import Any, Type
from importlib import import_module

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

def build_provider(name: str, **kwargs):
    cls = PROVIDER_MAP.get(name)

    if cls is None:
        from ai_cli.providers.bootstrap import init_providers
        init_providers()

        cls = PROVIDER_MAP.get(name)

    if cls is None:
        raise ValueError(f"Unknown provider: {name}")

    return cls(**kwargs)

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
