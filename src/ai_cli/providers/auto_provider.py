from __future__ import annotations

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.registry import PROVIDER_MAP


class AutoProvider:
    """
    Provider fallback wrapper.

    Tries providers in fallback_order until one succeeds.
    """

    def __init__(self, fallback_order=None):
        self.fallback_order = fallback_order or list(PROVIDER_MAP.keys())

    def send(self, prompt: str):
        errors = []

        for provider_name in self.fallback_order:
            provider_cls = PROVIDER_MAP.get(provider_name)

            if provider_cls is None:
                errors.append(f"{provider_name}: not found")
                continue

            try:
                provider = provider_cls()
                return provider.send(prompt)

            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")

        raise ProviderRequestError(
            f"Auto fallback exhausted. Errors: {'; '.join(errors)}"
        )