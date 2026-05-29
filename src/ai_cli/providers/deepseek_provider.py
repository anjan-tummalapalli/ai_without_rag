"""
deepseek_provider.py

DeepSeek provider implementation for AI Gateway / CLI applications.

This provider uses the OpenAI-compatible DeepSeek API.

Official docs:
 [oai_citation:0‡api-docs.deepseek.com](https://api-docs.deepseek.com/?utm_source=chatgpt.com)

Environment Variables:
    DEEPSEEK_API_KEY   -> Required API key
    DEEPSEEK_MODEL     -> Optional default model override

Default Models:
    deepseek-v4-flash
    deepseek-v4-pro

DeepSeek API is OpenAI-compatible, so this implementation
uses the official OpenAI Python SDK.
"""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


class DeepSeekProvider:
    """
    DeepSeek AI provider implementation.

    This class provides a simple wrapper around the
    DeepSeek OpenAI-compatible chat completion API.

    Attributes:
        api_key (str):
            DeepSeek API key.

        model (str):
            Default model used for requests.

        client (OpenAI):
            OpenAI-compatible client instance configured
            for DeepSeek.
    """

    DEFAULT_MODEL = "deepseek-v4-flash"
    BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """
        Initialize DeepSeek provider.

        Args:
            api_key:
                DeepSeek API key.
                If not provided, reads from:
                DEEPSEEK_API_KEY

            model:
                Optional model override.
                If not provided, reads from:
                DEEPSEEK_MODEL
                otherwise falls back to DEFAULT_MODEL.

        Raises:
            ValueError:
                If API key is missing.
        """

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not set")

        self.model = (
            model
            or os.getenv("DEEPSEEK_MODEL")
            or self.DEFAULT_MODEL
        )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    @property
    def provider_name(self) -> str:
        """
        Provider display name.

        Returns:
            str: Provider name.
        """

        return "deepseek"

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
        """
        Send a prompt to DeepSeek and return response text.

        Args:
            prompt:
                User input prompt.

            model:
                Optional per-request model override.

            temperature:
                Sampling temperature.
                Lower = deterministic.
                Higher = creative.

            max_tokens:
                Optional response token limit.

            system_prompt:
                Optional system instruction.

            timeout:
                Optional request timeout in seconds.

            **kwargs:
                Additional arguments forwarded to the API.

        Returns:
            str:
                Model response text.

        Raises:
            RuntimeError:
                If request fails.
        """

        selected_model = model or self.model

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        try:
            response = self.client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **kwargs,
            )

            content = response.choices[0].message.content

            return content.strip() if content else ""

        except Exception as exc:
            raise RuntimeError(
                f"DeepSeek request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Verify provider connectivity.

        Returns:
            bool:
                True if provider is reachable,
                otherwise False.
        """

        try:
            self.ask(
                prompt="Hello",
                max_tokens=5,
            )
            return True

        except Exception:
            return False