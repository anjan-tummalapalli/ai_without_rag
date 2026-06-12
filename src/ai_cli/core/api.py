"""Public API for sending prompts to AI providers."""

import os

from ai_cli.providers.registry import build_provider


def ask(
    prompt: str,
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    **kwargs,
):
    """Send a prompt to the given provider and return the response."""
    if api_key is None:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")
    ai_provider = build_provider(
        provider,
        model=model,
        api_key=api_key,
        timeout=timeout,
        **kwargs,
    )
    if hasattr(ai_provider, "send"):
        return ai_provider.send(prompt)
    return ai_provider.chat(prompt)
