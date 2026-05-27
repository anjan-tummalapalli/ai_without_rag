from __future__ import annotations
import inspect
import logging
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
        """
        Try providers in order and return the first successful response.

        Edge cases handled:
        - provider not registered
        - provider registered but not an AIProvider subclass (or missing send)
        - provider class that cannot be instantiated with (model=...)
        - provider does not support the requested model (if metadata present)
        - provider.send is async / returns coroutine (not supported here)
        - provider.send returns non-string (will be coerced to str)
        All failures are collected and surface in the raised ProviderRequestError.
        """
        errors: list[str] = []
        requested_model = getattr(self, "model", None)

        for provider_name in self.fallback_order:
            try:
                provider_entry = PROVIDER_MAP.get(provider_name)
                if provider_entry is None:
                    msg = f"{provider_name}: not registered"
                    errors.append(msg)
                    logger.warning("provider_fallback_skipped provider=%s reason=%s", provider_name, "not registered")
                    continue

                # If registry stores a provider class, try to instantiate it if it's a subclass of AIProvider.
                provider_instance = None
                if isinstance(provider_entry, type):
                    if issubclass(provider_entry, AIProvider):
                        try:
                            provider_instance = provider_entry(model=requested_model)
                        except Exception as exc:
                            msg = f"{provider_name}: failed to instantiate provider class: {exc}"
                            errors.append(msg)
                            logger.warning("provider_fallback_failed provider=%s error=%s", provider_name, str(exc))
                            continue
                    else:
                        msg = f"{provider_name}: registered entry is a class but not an AIProvider subclass"
                        errors.append(msg)
                        logger.warning("provider_fallback_skipped provider=%s reason=%s", provider_name, "invalid class type")
                        continue
                else:
                    provider_instance = provider_entry

                # Ensure provider has a callable send method
                send = getattr(provider_instance, "send", None)
                if not callable(send):
                    msg = f"{provider_name}: provider does not implement a callable send(prompt) method"
                    errors.append(msg)
                    logger.warning("provider_fallback_skipped provider=%s reason=%s", provider_name, "no send()")
                    continue

                # If provider has metadata, check if it supports the requested model.
                provider_meta = getattr(provider_instance, "provider_meta", None)
                if provider_meta is not None and requested_model:
                    supported = getattr(provider_meta, "supported_models", None)
                    if supported and requested_model not in supported and "auto" not in supported:
                        msg = (
                            f"{provider_name}: does not support model '{requested_model}' "
                            f"(supported: {supported})"
                        )
                        errors.append(msg)
                        logger.info("provider_fallback_skipped provider=%s reason=%s", provider_name, "model not supported")
                        continue

                # Call the provider
                try:
                    result = send(prompt)
                except Exception as exc:
                    # provider attempted to handle request but failed (e.g., missing API key, network error)
                    msg = f"{provider_name}: error during send(): {exc}"
                    errors.append(msg)
                    logger.warning("provider_fallback_failed provider=%s error=%s", provider_name, str(exc))
                    continue

                # If the provider returned a coroutine (async provider), this caller is sync -> skip
                if inspect.isawaitable(result):
                    msg = f"{provider_name}: send() returned an awaitable (async provider). AutoProvider._send_impl is synchronous and cannot await."
                    errors.append(msg)
                    logger.warning("provider_fallback_skipped provider=%s reason=%s", provider_name, "async send()")
                    continue

                # Coerce non-string responses to string, but treat None as an error
                if result is None:
                    msg = f"{provider_name}: send() returned None"
                    errors.append(msg)
                    logger.warning("provider_fallback_failed provider=%s error=%s", provider_name, "None response")
                    continue

                if not isinstance(result, str):
                    try:
                        response_text = str(result)
                    except Exception as exc:
                        msg = f"{provider_name}: failed to coerce response to string: {exc}"
                        errors.append(msg)
                        logger.warning("provider_fallback_failed provider=%s error=%s", provider_name, str(exc))
                        continue
                else:
                    response_text = result

                # Success
                return response_text

            except Exception as exc:
                # Catch-all for any unexpected issues per-provider to continue fallback loop
                error_msg = f"{provider_name}: unexpected error: {exc}"
                errors.append(error_msg)
                logger.exception("provider_fallback_unexpected provider=%s", provider_name)
                continue

        # If we get here, no provider succeeded.
        full_msg = (
            "Auto fallback exhausted. No provider succeeded. "
            "Checked providers in order: " + ", ".join(self.fallback_order) + ". "
            "Errors: " + "; ".join(errors)
        )
        raise ProviderRequestError(full_msg)