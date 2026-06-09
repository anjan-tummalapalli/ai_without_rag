from __future__ import annotations
from typing import Type, Dict, Any


class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, Type] = {}

    def register(self, name: str, provider_cls: Type):
        key = name.lower().strip()
        if key in self._providers:
            raise ValueError(f"Provider already registered: {key}")
        self._providers[key] = provider_cls

    def get(self, name: str) -> Type:
        key = name.lower().strip()
        if key not in self._providers:
            raise ValueError(f"Unknown provider: {name}")
        return self._providers[key]

    def all(self):
        return dict(self._providers)


# single instance (explicit global container, but deterministic)
registry = ProviderRegistry()


def build_provider(name: str, **kwargs):
    cls = registry.get(name)
    return cls(**kwargs)
