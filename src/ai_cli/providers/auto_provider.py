from __future__ import annotations
import logging
import os
from ai_cli.providers.base import AIProvider, ProviderMetadata
from ai_cli.providers.registry import PROVIDER_MAP
from ai_cli.core.exceptions import ProviderRequestError

logger = logging.getLogger("ai_gateway")

class AutoProvider(AIProvider):
    """Dynamically falls back to available providers based on API keys."""

    def __init__(self, model: str | None = None) -> None:
        meta = ProviderMetadata(
            name="Auto Fallback",
            default_model="auto",
            supported_models=["auto"],
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            max_context=128_000,
            cost_per_1k_tokens=0.01,
            avg_latency_ms=500,
        )
        super().__init__(provider_name="auto", model=model, provider_meta=meta)

        # Priority fallback order
        self.fallback_order = [
            "openai",
            "anthropic",
            "gemini",
            "perplexity",
            "cohere",
            "deepseek",
            "groq",
            "xai",
            "openrouter",
            "together",
            "fireworks",
        ]

def _send_impl(self, prompt: str) -> str:
    errors: list[str] = []
    for provider_name in self.fallback_order:
        try:
            provider = PROVIDER_MAP.get(provider_name)
            if provider is None:
                raise ProviderRequestError(f"Provider '{provider_name}' not registered")
            return provider.send(prompt)

        except Exception as exc:
            error_msg = f"{provider_name}: {str(exc)}"
            errors.append(error_msg)

            logger.warning(
                "provider_fallback_failed provider=%s error=%s",
                provider_name,
                str(exc),
            )
            # continue to next provider on error
            continue

    raise ProviderRequestError(
        f"Auto fallback exhausted. Errors: {errors}"
    )