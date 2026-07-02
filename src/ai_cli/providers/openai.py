from typing import Any

from ai_cli.providers.base import BaseProvider
from ai_cli.providers.decorators import chat_provider


@chat_provider("openai")
class OpenAIProvider(BaseProvider):
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key
        self.model = model

    def ask(self, prompt: str, **kwargs: Any) -> str:
        return f"OpenAI response: {prompt}"