"""
OpenAI ChatGPT provider implementation for ai_cli.
"""
from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import BaseProvider
from ai_cli.providers.registry import register_chat_provider, register_provider


class OpenAIProvider(BaseProvider):
    PROVIDER_NAME = "openai"
    DEFAULT_CHAT_MODEL = "gpt-5.5"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIProvider")
        if OpenAI is None:
            raise ProviderRequestError(
                "The 'openai' package is not installed. Install it via "
                "'pip install openai'."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or "gpt-4o-mini"

    def _send_impl(self, prompt: str) -> str:
        self._ensure_key()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise ProviderRequestError("OpenAI returned no choices")
            first = choices[0]
            message = getattr(first, "message", None) or (
                first.get("message") if isinstance(first, dict) else None
            )
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if not content:
                raise ProviderRequestError("OpenAI returned empty content")
            return content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(getattr(resp, "choices", None))
        except Exception:
            return False

    def send(self, prompt: str, **kwargs: Any) -> str:
        return self._send_impl(prompt)

    def ask(self, prompt: str, **kwargs: Any) -> str:
        return self.send(prompt, **kwargs)
    
    def _ensure_key(self):
        if not self.api_key and not os.getenv("OPENAI_API_KEY"):
            raise ProviderRequestError("OPENAI_API_KEY is required for OpenAIProvider")
    
    def is_ready(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

register_provider("openai", OpenAIProvider)
register_chat_provider("openai", OpenAIProvider)