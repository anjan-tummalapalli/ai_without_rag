"""Backwards-compatible ``ask()`` shim.

Existing callers that import ``ask`` from this module continue to work
unchanged.  Internally the call is now forwarded to ``AIService`` so all
retry, decoding, and provider-selection logic is kept in one place.
"""

from __future__ import annotations

import os
from typing import Any

from ai_cli.core.service.ai_service import AIService


def ask(
    prompt: str,
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> str:
    """Send *prompt* to *provider* and return the full response.

    Parameters
    ----------
    prompt:
        The message to send.
    provider:
        Provider name (default ``"auto"``).
    model:
        Optional model override.
    api_key:
        Explicit API key; falls back to the ``<PROVIDER>_API_KEY`` env var.
    timeout:
        Per-request timeout in seconds.

    Returns
    -------
    str
        Full AI response.
    """
    if api_key is None:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")

    # Pass api_key via the modules/profile path is not possible directly, but
    # AIService._build_kwargs uses inspect to forward only accepted kwargs.
    # For providers that accept api_key, we temporarily set it on the service
    # kwargs by subclassing the build step.  We do this by patching the
    # candidate dict after construction.
    svc = _AskServiceAdapter(
        provider=provider,
        model=model,
        timeout=int(timeout) if timeout is not None else 60,
        api_key=api_key,
    )
    return svc.ask(prompt)


class _AskServiceAdapter(AIService):
    """Internal subclass that injects ``api_key`` into the kwargs dict."""

    def __init__(self, *, api_key: str | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._api_key = api_key

    def _build_kwargs(self, prompt: str) -> dict[str, Any]:
        kwargs = super()._build_kwargs(prompt)
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        return kwargs
