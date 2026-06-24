"""AutoProvider: fallback wrapper that tries providers in sequence."""
from __future__ import annotations

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
        """Try each provider in fallback_order and return first success."""

        errors: list[str] = []

        for provider_name in self.fallback_order:
            provider_cls = PROVIDER_MAP.get(provider_name)

            if provider_cls is None:
                errors.append(f"{provider_name}: not found")
                continue

            try:
                provider = provider_cls()

            except ValueError as exc:
                msg = str(exc).lower()

                if "api key" in msg or "apikey" in msg or "api_key" in msg:
                    errors.append(
                        f"{provider_name}: skipped ({exc})"
                    )
                    continue

                errors.append(f"{provider_name}: init failed ({exc})")
                continue

            except Exception as exc:
                errors.append(f"{provider_name}: init failed ({exc})")
                continue

            try:
                return provider.send(prompt)

            except Exception as exc:
                msg = str(exc).lower()

                if any(
                    key in msg
                    for key in [
                        "401",
                        "unauthorized",
                        "403",
                        "forbidden",
                        "quota",
                        "authentication",
                        "invalid",
                        "api key",
                        "apikey",
                        "api_key",
                    ]
                ):
                    errors.append(
                        f"{provider_name}: skipped ({exc})"
                    )
                    continue

                errors.append(
                    f"{provider_name}: failed ({exc})"
                )
                continue

        raise ProviderRequestError(
            "Auto fallback exhausted. Errors:\n"
            + "\n".join(errors)
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