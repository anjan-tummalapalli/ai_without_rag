"""Minimal, dependency-free OpenAI provider used for simple/offline tests."""

from typing import Any

from ai_cli.providers.base import BaseProvider
from ai_cli.providers.decorators import chat_provider


@chat_provider("openai")
class OpenAIProvider(BaseProvider):
    """Lightweight stand-in that echoes the prompt without calling any API."""

    def __init__(
        self, api_key: str | None = None, model: str | None = None
    ) -> None:
        super().__init__(api_key=api_key, model=model)

    def send(self, prompt: str, **kwargs: Any) -> str:
        """Return a deterministic response without contacting OpenAI."""
        return f"OpenAI response: {prompt}"
