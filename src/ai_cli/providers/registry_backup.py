from __future__ import annotations

from typing import Type, Dict, Any, Callable

PROVIDER_MAP: dict[str, Type] = {}
CHAT_PROVIDERS: dict[str, Type] = {}
EMBEDDING_PROVIDERS: dict[str, Type] = {}

# -----------------------------
# Internal registries
# -----------------------------
_PROVIDER_REGISTRY: Dict[str, Type] = {}
_CHAT_PROVIDER_REGISTRY: Dict[str, Type] = {}
_EMBED_PROVIDER_REGISTRY: Dict[str, Type] = {}


# -----------------------------
# Registration API
# -----------------------------
def register_provider(name: str, cls: Type) -> None:
    _PROVIDER_REGISTRY[name] = cls


def register_chat_provider(name: str, cls: Type) -> None:
    _CHAT_PROVIDER_REGISTRY[name] = cls


def register_embedding_provider(name: str, cls: Type) -> None:
    _EMBED_PROVIDER_REGISTRY[name] = cls


# -----------------------------
# Resolution API
# -----------------------------
def build_provider(name: str, **kwargs: Any):
    cls = _PROVIDER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown provider: {name}")
    return cls(**kwargs)


def get_chat_provider(name: str, **kwargs: Any):
    cls = _CHAT_PROVIDER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown chat provider: {name}")
    return cls(**kwargs)


def get_embedding_provider(name: str, **kwargs: Any):
    cls = _EMBED_PROVIDER_REGISTRY.get(name)
    if not cls:
        raise ValueError(f"Unknown embedding provider: {name}")
    return cls(**kwargs)


# -----------------------------
# Safe introspection (useful for CLI/debug)
# -----------------------------
def list_providers():
    return sorted(_PROVIDER_REGISTRY.keys())

# -----------------------------
# Backward compatibility layer
# (required for existing tests)
# -----------------------------
CHAT_PROVIDERS = _CHAT_PROVIDER_REGISTRY
PROVIDER_MAP = _PROVIDER_REGISTRY

def register_embedding_provider(name: str, cls):
    EMBEDDING_PROVIDERS[name] = cls
    PROVIDER_MAP[name] = cls

def get_embedding_provider(name: str, **kwargs):
    cls = EMBEDDING_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown embedding provider: {name}")
    return cls(**kwargs)
