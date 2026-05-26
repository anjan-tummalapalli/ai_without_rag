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
        errors = []
        for p_name in self.fallback_order:
            if p_name not in PROVIDER_MAP:
                continue

            provider_class = PROVIDER_MAP[p_name]
            # Simple check: Does it have an env variable requirement?
            if hasattr(provider_class, "api_key_env"):
                env_var = provider_class.api_key_env
                if not os.getenv(env_var):
                    continue
            elif p_name == "openai" and not os.getenv("OPENAI_API_KEY"):
                continue
            elif p_name == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
                continue

            logger.info("AutoProvider attempting to use %s", p_name)

            try:
                # Instantiate and route
                instance = provider_class(model=None)  # use their default model
                response = instance.send(prompt)

                # Copy metrics over
                self.metrics.total_prompt_tokens += (
                    instance.metrics.total_prompt_tokens
                )
                self.metrics.total_completion_tokens += (
                    instance.metrics.total_completion_tokens
                )

                return response
            except Exception as exc:
                logger.warning("AutoProvider %s failed: %s", p_name, exc)
                errors.append(f"{p_name}: {exc}")

        raise ProviderRequestError(f"Auto fallback exhausted. Errors: {errors}")