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
        """Try each provider in fallback_order and return the first success."""

        errors: list[str] = []

        for provider_name in self.fallback_order:
            provider_cls = PROVIDER_MAP.get(provider_name)

            if provider_cls is None:
                errors.append(f"{provider_name}: not found")
                continue

            is_configured = getattr(provider_cls, "is_configured", None)

            if callable(is_configured):
                try:
                    if not is_configured():
                        errors.append(
                            f"{provider_name}: skipped (not configured)"
                        )
                        continue
                except Exception:
                    pass

            try:
                provider = provider_cls()

            except ValueError as exc:
                msg = str(exc).lower()

                if "api key" in msg or "apikey" in msg:
                    errors.append(
                        f"{provider_name}: skipped ({exc})"
                    )
                    continue

                raise

            except Exception as exc:
                errors.append(
                    f"{provider_name}: init failed ({exc})"
                )
                continue

            try:
                return provider.send(prompt)

            except ProviderRequestError as exc:
                errors.append(
                    f"{provider_name}: {exc}"
                )
                continue

            except ValueError as exc:
                msg = str(exc).lower()

                if "api key" in msg or "apikey" in msg:
                    errors.append(
                        f"{provider_name}: skipped ({exc})"
                    )
                    continue

                raise

            except Exception:
                raise

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