"""AutoProvider: fallback wrapper that tries providers in sequence."""
from __future__ import annotations

import os
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.registry import PROVIDER_MAP


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
        self.fallback_order = fallback_order or [
            name for name in default_order if name in PROVIDER_MAP
        ]

    def send(self, prompt: str) -> str:
        """Try each provider in fallback_order and return the first success."""

        errors: list[str] = []

        for provider_name in self.fallback_order:
            provider_cls = PROVIDER_MAP.get(provider_name)

            if provider_cls is None:
                errors.append(f"{provider_name}: not found")
                continue

            try:
                provider = provider_cls()

                return provider.send(prompt)

            except ValueError as exc:
                # Missing API keys/config should skip this provider
                # but allow mock/test providers to work.
                message = str(exc)

                if "API_KEY" in message or "api key" in message.lower():
                    errors.append(f"{provider_name}: skipped ({message})")
                    continue

                errors.append(f"{provider_name}: {exc}")
                continue

            except ProviderRequestError as exc:
                errors.append(f"{provider_name}: {exc}")

        raise ProviderRequestError(
            f"Auto fallback exhausted. Errors: {'; '.join(errors)}"
        )

    def ask(self, prompt: str, **_kwargs) -> str:
        """Chat-compatible interface; delegates to send().

        Args:
            prompt: User input to send.
            **_kwargs: Ignored; present for provider contract compatibility.

        Returns:
            Model response string.
        """
        return self.send(prompt)