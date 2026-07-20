"""Shared AI service layer.

``AIService`` is the single backend that both the CLI and any GUI should use
to interact with AI providers.  It owns:

* Provider construction via ``core.api.ask``
* Exponential-backoff retry on transient errors
* Chunk decoding for bytes / str / arbitrary objects
* Sync, streaming (``Iterator``), and async (``AsyncIterator``) response paths

Typical usage
-------------
CLI (one-shot)::

    svc = AIService(provider="openai", model="gpt-4o-mini", timeout=60)
    reply = svc.ask("What is Python?")

GUI (session-scoped instance, streaming)::

    svc = AIService(provider="gemini", stream=True)
    for chunk in svc.ask_stream("Explain async/await"):
        widget.append(chunk)

Async GUI (e.g. asyncio-based)::

    svc = AIService(provider="openai")
    async for chunk in svc.ask_async("Tell me a joke"):
        await update_ui(chunk)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from ai_cli.core.api import ask as _core_ask

logger = logging.getLogger("ai_cli.service")

# Hard ceiling on backoff sleep to avoid accidental DoS-in-place.
_MAX_BACKOFF_SECONDS: float = 30.0


def _decode_chunk(chunk: Any) -> str:
    """Decode a streaming chunk to a UTF-8 string.

    Handles ``bytes``, ``str``, and arbitrary objects (JSON-serialised as a
    fallback so callers always receive a ``str``).
    """
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    if isinstance(chunk, str):
        return chunk
    try:
        return json.dumps(chunk, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(chunk)


class AIService:
    """Provider-agnostic AI service shared by the CLI and GUI.

    Parameters
    ----------
    provider:
        Provider name (default ``"auto"``).
    model:
        Optional model override.
    timeout:
        Per-request timeout in seconds (default ``60``).
    profile:
        Optional named profile forwarded to the provider.
    stream:
        When ``True``, ``ask()`` requests a streaming response; chunks are
        concatenated into the returned string.  Prefer ``ask_stream()`` or
        ``ask_async()`` when you need incremental output.
    modules:
        Comma-separated module names to enable (passed through to the
        underlying ``ask()`` call when accepted).
    max_retries:
        Maximum number of attempts on transient errors (default ``3``).
    backoff:
        Initial backoff in seconds for the exponential-retry strategy
        (default ``0.5``).
    """

    def __init__(
        self,
        *,
        provider: str = "auto",
        model: str | None = None,
        timeout: int = 60,
        profile: str | None = None,
        stream: bool = False,
        modules: str | None = None,
        max_retries: int = 3,
        backoff: float = 0.5,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")

        self.provider = (
            provider.strip() if provider and provider.strip() else "auto"
        )
        self.model = (
            model.strip() if isinstance(model, str) and model.strip() else None
        )
        self.timeout = timeout
        self.profile = (
            profile.strip()
            if isinstance(profile, str) and profile.strip()
            else None
        )
        self.stream = stream
        self.modules: list[str] | None = None
        if isinstance(modules, str) and modules.strip():
            self.modules = [m.strip() for m in modules.split(",") if m.strip()]
        self.max_retries = max_retries
        self.backoff = backoff

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kwargs(self, prompt: str) -> dict[str, Any]:
        """Build kwargs for ``core.api.ask``, filtering to accepted params."""
        candidate: dict[str, Any] = {
            "provider": self.provider,
            "prompt": prompt,
            "model": self.model,
            "timeout": self.timeout,
            "profile": self.profile,
            "stream": self.stream,
            "modules": self.modules,
        }

        try:
            sig = inspect.signature(_core_ask)
            params = sig.parameters
            accepts_var_kw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            filtered: dict[str, Any] = {
                k: v
                for k, v in candidate.items()
                if v is not None and (accepts_var_kw or k in params)
            }
            filtered.setdefault("prompt", prompt)
            filtered.setdefault("provider", self.provider)
            return filtered
        except (TypeError, ValueError):
            logger.debug(
                "Failed to inspect core ask() signature; sending best-effort args"
            )
            base: dict[str, Any] = {
                "provider": self.provider,
                "prompt": prompt,
                "timeout": self.timeout,
            }
            if self.model is not None:
                base["model"] = self.model
            if self.profile is not None:
                base["profile"] = self.profile
            if self.stream:
                base["stream"] = self.stream
            if self.modules is not None:
                base["modules"] = self.modules
            return base

    @staticmethod
    async def _drain_async(result: Any) -> str:
        """Drain an awaitable / async-iterable result into a single string."""
        value = await result if inspect.isawaitable(result) else result
        parts: list[str] = []
        if isinstance(value, AsyncIterator):
            async for part in value:
                parts.append(_decode_chunk(part))
        elif hasattr(value, "__aiter__"):
            async for part in value:
                parts.append(_decode_chunk(part))
        elif hasattr(value, "__iter__") and not isinstance(
            value, str | bytes | dict
        ):
            for part in value:
                parts.append(_decode_chunk(part))
        else:
            parts.append(_decode_chunk(value))
        return "".join(parts)

    @staticmethod
    def _drain_sync(result: Any) -> str:
        """Drain a synchronous (possibly iterable) result into a single string."""
        if inspect.isawaitable(result) or hasattr(result, "__aiter__"):
            return asyncio.run(AIService._drain_async(result))
        parts: list[str] = []
        if hasattr(result, "__iter__") and not isinstance(
            result, str | bytes | dict
        ):
            for part in result:
                parts.append(_decode_chunk(part))
        elif isinstance(result, dict | list):
            parts.append(
                json.dumps(result, indent=2, ensure_ascii=False, default=str)
            )
        else:
            parts.append(_decode_chunk(result))
        return "".join(parts)

    def _call_with_retries(self, prompt: str) -> Any:
        """Call ``core.api.ask`` with exponential-backoff retries.

        Returns the raw result from ``ask()`` on success, or raises on
        terminal failure.
        """
        kwargs = self._build_kwargs(prompt)

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "AIService ask() attempt %d/%d provider=%s",
                    attempt,
                    self.max_retries,
                    self.provider,
                )
                return _core_ask(**kwargs)
            except (TimeoutError, ConnectionError, OSError) as exc:
                logger.warning(
                    "Transient error on attempt %d: %s", attempt, exc
                )
                if attempt == self.max_retries:
                    logger.error(
                        "Max retries (%d) reached. Failing.", self.max_retries
                    )
                    raise
                sleep_secs = min(
                    self.backoff * (2 ** (attempt - 1)), _MAX_BACKOFF_SECONDS
                )
                time.sleep(sleep_secs)

        # Unreachable — satisfies type checkers.
        raise RuntimeError(
            "AIService._call_with_retries exhausted unexpectedly"
        )  # pragma: no cover

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, prompt: str) -> str:
        """Send *prompt* to the configured provider and return the full reply.

        Collects streaming chunks into a single string when ``stream=True``.
        Retries on transient network/timeout errors up to ``max_retries``
        times with exponential backoff.

        Parameters
        ----------
        prompt:
            The message to send.

        Returns
        -------
        str
            Complete AI response as a string.

        Raises
        ------
        TimeoutError, ConnectionError, OSError
            Propagated after all retry attempts are exhausted.
        RuntimeError, TypeError, ValueError
            Non-transient errors from the provider.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")

        result = self._call_with_retries(prompt)
        return self._drain_sync(result)

    def ask_stream(self, prompt: str) -> Iterator[str]:
        """Send *prompt* and yield response chunks one at a time (sync).

        Suitable for CLI output or Tkinter GUIs running on the main thread.
        Uses the underlying ``stream=True`` flag regardless of the instance's
        default ``self.stream`` setting.

        Parameters
        ----------
        prompt:
            The message to send.

        Yields
        ------
        str
            Individual decoded response chunks.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")

        # Temporarily force stream on for this call
        original_stream = self.stream
        self.stream = True
        try:
            result = self._call_with_retries(prompt)
        finally:
            self.stream = original_stream

        if inspect.isawaitable(result) or hasattr(result, "__aiter__"):
            # Provider returned an async iterable — drain it synchronously
            full = asyncio.run(self._drain_async(result))
            yield full
            return

        if hasattr(result, "__iter__") and not isinstance(
            result, str | bytes | dict
        ):
            for part in result:
                yield _decode_chunk(part)
        else:
            yield _decode_chunk(result)

    async def ask_async(self, prompt: str) -> AsyncIterator[str]:
        """Send *prompt* and yield response chunks asynchronously.

        Suitable for asyncio-based GUIs or frameworks (e.g. FastAPI, aiohttp,
        Textual).

        Parameters
        ----------
        prompt:
            The message to send.

        Yields
        ------
        str
            Individual decoded response chunks.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")

        # Run the blocking provider call in a thread pool so we don't block the
        # event loop.
        loop = asyncio.get_event_loop()
        original_stream = self.stream
        self.stream = True
        try:
            result = await loop.run_in_executor(
                None, self._call_with_retries, prompt
            )
        finally:
            self.stream = original_stream

        if inspect.isawaitable(result):
            result = await result

        if hasattr(result, "__aiter__"):
            async for part in result:
                yield _decode_chunk(part)
        elif hasattr(result, "__iter__") and not isinstance(
            result, str | bytes | dict
        ):
            for part in result:
                yield _decode_chunk(part)
        else:
            yield _decode_chunk(result)

    def __repr__(self) -> str:
        return (
            f"AIService(provider={self.provider!r}, model={self.model!r}, "
            f"timeout={self.timeout}, stream={self.stream}, "
            f"max_retries={self.max_retries})"
        )


__all__ = ["AIService"]
