import os
from typing import Any, cast

from openai import OpenAI

from ai_cli.providers.base import BaseProvider


class PerplexityProvider(BaseProvider):
    def __init__(
        self,
        model: str = "sonar",
        api_key: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")

        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY is required")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.perplexity.ai",
        )

    def _send_impl(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=cast(str, self.model),
            messages=[{"role": "user", "content": prompt}],
        )
        choices = getattr(response, "choices", [])
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        if not message:
            return ""
        content = getattr(message, "content", "")
        return content.strip() if content else ""

    def send(self, prompt: str, **kwargs: Any) -> str:
        return self._send_impl(prompt)

    def ask(self, prompt: str, **kwargs: Any) -> str:
        return self.send(prompt, **kwargs)
