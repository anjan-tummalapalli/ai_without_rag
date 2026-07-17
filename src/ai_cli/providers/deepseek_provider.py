"""
deepseek_provider.py

DeepSeek provider implementation for AI Gateway / CLI apps.

This provider uses the OpenAI-compatible DeepSeek API.

Added:
 - embeddings() helper to create embeddings via the
     OpenAI-compatible embeddings endpoint.
 - small docs for RAG workflows that use chat + embeddings.

Environment Variables:
      DEEPSEEK_API_KEY   -> Required API key
      DEEPSEEK_MODEL     -> Optional default chat model
      DEEPSEEK_EMBEDDING_MODEL -> Optional default embedding
                                                         model override

Default Models:
      deepseek-v4-flash
      deepseek-v4-pro
      text-embedding-3-small (for embeddings)
"""

from __future__ import annotations

import os
from typing import Any, cast

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENAI_AVAILABLE = False

from ai_cli.providers.base import BaseProvider
from ai_cli.providers.registry import (
    register_chat_provider,
    register_provider,
)


class DeepSeekProvider(BaseProvider):
    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_EMBED_MODEL = "text-embedding-3-small"
    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        super().__init__(model=self.DEFAULT_MODEL)
        if api_key == "":
            raise ValueError("DEEPSEEK_API_KEY not set")

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "The 'openai' package is not installed."
            )

        self.client: Any | None = None

        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.BASE_URL,
            )

    def health_check(self) -> bool:
        if not self.api_key:
            return False

        try:
            self.ask(
                "ping",
                max_tokens=1,
            )
            return True

        except Exception:
            return False

    def ask(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> str:
        selected_model = model or self.DEFAULT_MODEL

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            if self.client is None:
                raise RuntimeError("DeepSeek client is not initialized.")

            client = self.client
            response = client.chat.completions.create(
                model=selected_model,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )
            content = response.choices[0].message.content

            if content is None:
                return ""

            return str(content).strip()
        except Exception as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

    def embeddings(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        """
        Create embeddings for a list of texts.

        Returns a list of float vectors corresponding to each input text.
        """
        selected = model or self.DEFAULT_EMBED_MODEL
        try:
            if self.client is None:
                raise RuntimeError("DeepSeek client is not initialized.")
            response: Any = self.client.embeddings.create(
                model=selected, input=texts
            )
            vectors: list[list[float]] = []

            for item in response.data:
                vectors.append(cast(list[float], item.embedding))

            return vectors
        except Exception as exc:
            raise RuntimeError(
                f"DeepSeek embedding request failed: {exc}"
            ) from exc

    def _chat(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Any:
        if self.client is None:
            raise RuntimeError("DeepSeek client is not initialized.")

        return self.client.chat.completions.create(
            model=self.DEFAULT_MODEL,
            messages=cast(
                Any,
                [{"role": "user", "content": prompt}],
            ),
            **kwargs,
        )

    def send(self, prompt: str, **kwargs: Any) -> str:
        try:
            if self.client is None:
                raise RuntimeError("DeepSeek client is not initialized.")

            response = self._chat(prompt, **kwargs)

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]

                message = getattr(choice, "message", None)

                if isinstance(message, dict):
                    return str(message.get("content", ""))

                if message:
                    content = getattr(message, "content", None)
                    if content:
                        return str(content)

                text = getattr(choice, "text", None)
                if text:
                    return str(text)

            if isinstance(response, str):
                return response

            return str(response)

        except Exception as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

    def chat(self, prompt: str, **kwargs: Any) -> str:
        try:
            response = self._chat(prompt, **kwargs)

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]

                if hasattr(choice, "message"):
                    content = getattr(choice.message, "content", None)
                    if content:
                        return str(content)

                if hasattr(choice, "text"):
                    return str(choice.text)

            return str(response)

        except Exception as exc:
            raise RuntimeError(f"DeepSeek connection failed: {exc}") from exc


register_provider("deepseek", DeepSeekProvider)

register_chat_provider("deepseek", DeepSeekProvider)