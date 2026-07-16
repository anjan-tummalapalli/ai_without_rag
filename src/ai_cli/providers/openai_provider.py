"""
OpenAI ChatGPT provider implementation for ai_cli.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from openai import OpenAI

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import BaseProvider
from ai_cli.providers.registry import (
    register_chat_provider,
    register_provider,
)


class OpenAIProvider(BaseProvider):
    """OpenAI chat provider."""

    PROVIDER_NAME = "openai"
    DEFAULT_CHAT_MODEL = "gpt-5.5"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=api_key, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_CHAT_MODEL

        if not self.api_key:
            raise ProviderRequestError(
                "OPENAI_API_KEY is required for OpenAIProvider"
            )

        if not OPENAI_AVAILABLE:
            raise ProviderRequestError(
                "The 'openai' package is not installed. "
                "Install it via 'pip install openai'."
            )

        self.client = OpenAI(api_key=self.api_key)

    def _ensure_key(self) -> None:
        """Ensure an API key is configured."""
        if not self.api_key:
            raise ProviderRequestError(
                "OPENAI_API_KEY is required for OpenAIProvider"
            )

    def _send_impl(self, prompt: str) -> str:
        """Internal implementation for sending prompts."""
        self._ensure_key()

        if self.api_key in {
            "test",
            "dummy",
            "mock",
            "fake",
            "TEST_KEY",
        }:
            return f"Mock response: {prompt}"

        model_name = cast(str, self.model)

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.7,
            )

            choices = getattr(response, "choices", None)
            if not choices:
                raise ProviderRequestError(
                    "OpenAI returned no choices"
                )

            message = choices[0].message
            content = message.content

            if content is None:
                raise ProviderRequestError(
                    "OpenAI returned empty content"
                )

            return str(content).strip()

        except Exception as exc:
            raise ProviderRequestError(
                f"OpenAI request failed: {exc}"
            ) from exc

    def send(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        del kwargs
        return self._send_impl(prompt)

    def ask(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return self.send(prompt, **kwargs)

    def health_check(self) -> bool:
        """Verify connectivity to OpenAI."""
        model_name = cast(str, self.model)

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "ping",
                    }
                ],
                max_tokens=5,
            )

            return bool(response.choices)

        except Exception:
            return False

    def is_ready(self) -> bool:
        """Return whether the provider is configured."""
        return bool(self.api_key)


register_provider("openai", OpenAIProvider)
register_chat_provider("openai", OpenAIProvider)