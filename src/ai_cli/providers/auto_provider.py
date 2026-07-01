"""AutoProvider: fallback wrapper that tries providers in sequence."""
from __future__ import annotations

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers import registry

# compatibility alias for tests/plugins that monkeypatch this module
PROVIDER_MAP = registry.PROVIDER_MAP


class AutoProvider:
    """Provider fallback wrapper.

    Tries providers in fallback_order until one succeeds.
    """

    def __init__(
        self,
        fallback_order: list | None = None,
        **_kwargs,
    ):
        default_order = [
            "openai",
            "gemini",
            "cohere",
            "perplexity",
            "xai",
            "zai",
            "echo",
        ]

        registry.ensure_initialized()

        if fallback_order is not None:
            self.fallback_order = fallback_order
        else:
            self.fallback_order = [
                name
                for name in default_order
                if name in registry.PROVIDER_MAP
            ]

    def send(self, prompt: str) -> str:
        """Try each provider in fallback_order."""

        errors: list[str] = []

        for provider_name in self.fallback_order:

            provider_cls = PROVIDER_MAP.get(provider_name)

            if provider_cls is None:
                provider_cls = registry.PROVIDER_MAP.get(provider_name)

            if provider_cls is None:
                errors.append(
                    f"{provider_name}: not found"
                )
                continue

            try:
                provider = provider_cls()

            except Exception as exc:
                errors.append(
                    f"{provider_name}: init failed ({exc})"
                )
                continue

            try:
                return provider.send(prompt)

            except Exception as exc:
                errors.append(
                    f"{provider_name}: failed ({exc})"
                )
                continue

        raise ProviderRequestError(
            "Auto fallback exhausted. Errors:\n"
            + "\n".join(errors)
        )


    def ask(self, prompt: str, **_kwargs) -> str:
        return self.send(prompt)