from __future__ import annotations
import logging

from ai_cli.providers.registry import build_provider, load_plugins, PROVIDERS
from ai_cli.core.exceptions import AIProviderError

logger = logging.getLogger("ai_gateway")

# Ensure plugins are loaded at import time
load_plugins()


def ask(
    *,
    provider: str,
    prompt: str,
    model: str | None = None,
    timeout: float | None = 30.0,
) -> str:
    """High-level convenience function to ask a provider a prompt."""
    if not isinstance(provider, str):
        return "[ERROR] provider must be string"
    provider = provider.strip().lower()
    if not provider:
        return "[ERROR] provider is empty"

    if provider not in ("auto", "echo") and provider not in PROVIDERS:
        available = ", ".join(sorted(list(PROVIDERS.keys()) + ["auto", "echo"]))
        return (
            f"[ERROR] Invalid provider '{provider}'. "
            f"Available providers: {available}"
        )

    if not isinstance(prompt, str):
        return "[ERROR] prompt must be string"
    prompt = prompt.strip()
    if not prompt:
        return "[ERROR] Invalid prompt"

    if model is not None and provider != "auto":
        if not isinstance(model, str):
            return "[ERROR] model must be string"
        model = model.strip()
        if not model:
            return "[ERROR] model is empty"
        supported_models = PROVIDERS[provider].supported_models
        if model not in supported_models:
            supported = ", ".join(supported_models)
            return (
                f"[ERROR] Invalid model '{model}' for provider "
                f"'{provider}'. Supported models: {supported}"
            )

    try:
        ai_provider = build_provider(name=provider, model=model)

        # We override timeout via the base class if provided
        if timeout:
            ai_provider.timeout = int(timeout)

        response = ai_provider.send(prompt)

        if not isinstance(response, str):
            logger.error(
                "invalid_response_type provider=%s type=%s",
                provider,
                type(response).__name__,
            )
            return "[ERROR] Invalid response type"
        response = response.strip()
        if not response:
            return "[ERROR] Empty response"
        return response
    except AIProviderError as exc:
        logger.error("ai_provider_error provider=%s error=%s", provider, exc)
        return f"[ERROR] {exc}"
    except Exception as exc:
        logger.exception(
            "unexpected_ask_failure provider=%s model=%s error=%s",
            provider,
            model,
            exc,
        )
        return "[ERROR] Unexpected internal error. Check logs."
