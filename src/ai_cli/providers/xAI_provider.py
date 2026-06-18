"""
xAI Grok provider implementation for ai_cli.

This module integrates xAI Grok models into the ai_cli provider framework using
xAI's OpenAI-compatible API.

Environment Variables
---------------------
XAI_API_KEY
    API key used to authenticate with xAI API.

Example
-------
export XAI_API_KEY="your_api_key"

Usage
-----
provider = XAIProvider(
    model="grok-2-latest"
)

response = provider.send("Explain Kubernetes operators")
"""

import logging

from openai.types.chat import ChatCompletionMessageParam  # noqa: E402

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

from ai_cli.core.exceptions import ProviderRequestError

from .base import AIProvider


class InMemoryVectorStore:

    """Compatibility placeholder for exported vector store alias."""

    pass

logger = logging.getLogger(__name__)

class XAIProvider(AIProvider):
    """
    AI provider implementation for xAI Grok models.

    This provider communicates with xAI's OpenAI-compatible chat completions APIs.
    """

    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            provider_name="xai",
            model=model or "grok-2-latest",
            api_key=api_key,
            **kwargs,
        )
        if OpenAI is None:
            raise ProviderRequestError(
                "openai package is required. Install with `pip install openai`"
            )
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    def _call_model(self, prompt: str, system_prompt: str | None = None) -> str:
        """
        Internal helper: send prompt (with optional system message) to the Grok model.
        """
        try:
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                messages=messages, model=self.model, temperature=0.7
            )

            if not getattr(response, "choices", None):
                raise ProviderRequestError("xAI returned no completion choices")
            message = response.choices[0].message

            if not message or not getattr(message, "content", None):
                return "[No response from xAI]"
            return message.content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"xAI request failed: {exc}") from exc

    def _send_impl(self, prompt: str) -> str:
        """Send prompt to Grok model (required by AIProvider base class)."""
        try:
            return self._call_model(prompt)
        except ProviderRequestError as e:
            logger.warning(f"XAIProvider encountered an error: {e}")
            return "[Error: unable to get response]"

    def health_check(self) -> bool:
        """
        Perform lightweight connectivity test.

        Returns True if provider is operational, otherwise False.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "ping"},
                ],
                max_tokens=5,
            )
            return bool(getattr(response, "choices", None))
        except Exception:
            return False

    def send(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )

            if not getattr(response, "choices", None):
                return "[Error: unable to get response]"
            content = response.choices[0].message.content
            if not content:
                return "[Error: unable to get response]"
            return content

        except Exception as exc:
            raise ProviderRequestError(f"xAI request failed: {exc}") from exc


__all__ = ["XAIProvider", "InMemoryVectorStore"]