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
    """
    Module: ai_cli.core.api
    Purpose:
        Provide a small API surface for sending prompts to configured AI providers.
        This module exposes a single high-level convenience function `ask(...)`
        that validates inputs, selects/builds the requested provider, forwards the
        prompt, and returns a plain string with the provider's response.

    End result:
        On success, `ask` returns the provider's response as a stripped string.
        On any validation or runtime error the function returns a user-facing
        error string beginning with "[ERROR]" and logs details for diagnostics.

    Ask a configured AI provider a prompt and return the provider's textual response.

    Behavior summary:
    - Validates `provider`, `prompt`, and (optionally) `model`.
    - Normalizes `provider` by stripping and lowercasing it.
    - Allows special provider names "auto" and "echo" in addition to keys in PROVIDERS.
    - If a specific `model` is supplied and `provider != "auto"`, the model is validated
      against PROVIDERS[provider].supported_models.
    - Builds the provider instance via build_provider(name=provider, model=model).
    - If `timeout` is provided and truthy, overrides the provider instance timeout
      by setting ai_provider.timeout = int(timeout).
    - Sends the prompt via ai_provider.send(prompt).
    - If the provider returns a non-string or an empty string, returns an appropriate
      "[ERROR]" message.
    - Catches AIProviderError and general Exception, logs them, and returns
      user-facing "[ERROR]" messages rather than raising.

    Parameters:
        provider (str):
            The name of the provider to use. This will be stripped and lowercased.
            Allowed values: "auto", "echo", or any key present in the global
            PROVIDERS mapping. If invalid or empty, the function returns
            "[ERROR] ..." describing available providers.
        prompt (str):
            The prompt to send to the provider. Must be a non-empty string after
            stripping; otherwise the function returns "[ERROR] Invalid prompt".
        model (Optional[str], default=None):
            Optional model identifier. If provided and provider != "auto", it must be
            a non-empty string and one of the provider's supported_models; otherwise
            an "[ERROR] Invalid model ..." message is returned. When provider == "auto"
            the model parameter is ignored.
        timeout (Optional[float], default=30.0):
            Optional timeout in seconds. If truthy, the value is converted to int
            and assigned to the selected provider instance's timeout attribute before
            sending the prompt.

    Returns:
        str:
            On success: the provider's response, stripped of leading/trailing whitespace.
            On error: a string beginning with "[ERROR]" describing the failure. Examples:
              - "[ERROR] provider must be string"
              - "[ERROR] Invalid provider 'x'. Available providers: auto, echo, ... "
              - "[ERROR] Invalid model 'm' for provider 'p'. Supported models: ..."
              - "[ERROR] Empty response"
              - "[ERROR] Unexpected internal error. Check logs."

    Side effects:
    - Calls build_provider(...) and ai_provider.send(...).
    - May set ai_provider.timeout if a timeout value is provided.
    - Logs validation errors, provider-specific errors (AIProviderError), and unexpected
      exceptions via the module logger for debugging/telemetry.

    Notes:
    - The function intentionally returns error messages as strings (prefixed with
      "[ERROR]") rather than raising exceptions to keep a simple textual CLI-style
      contract for callers.
    - The exact semantics for provider/model discovery depend on the global PROVIDERS
      mapping and the build_provider implementation.
    """
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
