from typing import Any

from ai_cli.providers.config import resolve_api_key
from ai_cli.providers.loader import load_all_providers
from ai_cli.providers.resolver import resolve_provider_name

_PROVIDERS = load_all_providers()


def build_provider(request: Any) -> Any:
    provider_name = resolve_provider_name(request.provider)

    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")

    cls = _PROVIDERS[provider_name]

    api_key = resolve_api_key(provider_name, request.api_key)

    kwargs = request.kwargs or {}
    kwargs["api_key"] = api_key

    return cls(provider_name=provider_name, model=request.model, **kwargs)
