from typing import Type
from ai_cli.providers.contracts import (
    ChatProvider,
    EmbeddingProvider,
    RAGProvider,
)

CHAT_PROVIDERS: dict[str, Type] = {}
EMBEDDING_PROVIDERS: dict[str, Type] = {}
RAG_PROVIDERS: dict[str, Type] = {}
# Legacy compatibility
PROVIDER_MAP: dict[str, Type] = {}
PROVIDERS = PROVIDER_MAP

def register_chat_provider(name: str):
    def decorator(cls):
        CHAT_PROVIDERS[name] = cls
        PROVIDER_MAP[name] = cls
        return cls
    return decorator

def register_embedding_provider(name: str):
    def decorator(cls):
        EMBEDDING_PROVIDERS[name] = cls
        return cls
    return decorator

def register_rag_provider(name: str):
    def decorator(cls):
        RAG_PROVIDERS[name] = cls
        PROVIDER_MAP[name] = cls
        return cls
    return decorator

def build_provider(name: str, **kwargs):
    cls = PROVIDER_MAP.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name}")
    return cls(**kwargs)

def get_chat_provider(name: str, **kwargs):
    cls = CHAT_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown chat provider: {name}")
    return cls(**kwargs)

def load_plugins() -> None:
    """
    Backward compatibility shim.
    Providers are registered during module import,
    so no plugin loading is required.
    """
    return None
