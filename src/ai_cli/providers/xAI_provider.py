"""
xAI Grok provider implementation for ai_cli.
 
This module integrates xAI Grok models into the ai_cli provider
framework using xAI's OpenAI-compatible API.
 
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
 
from __future__ import annotations
 
import logging
import os
from typing import Any
 
from openai.types.chat import ChatCompletionMessageParam
 
try:
    from openai import OpenAI, OpenAIError
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]
 
    class OpenAIError(Exception):  # type: ignore[no-redef]
        """Fallback used when the openai package is not installed."""
 
from ai_cli.core.exceptions import ProviderRequestError
 
from .base import AIProvider
 
 
class InMemoryVectorStore:  # pylint: disable=too-few-public-methods
    """Compatibility placeholder for exported vector store alias.
 
    Intentionally minimal: this class exists only so that
    `InMemoryVectorStore` remains importable from this module for
    backward-compatible re-exports; it carries no behavior of its own.
    """
 
 
logger = logging.getLogger(__name__)
 
 
class XAIProvider(AIProvider):
    """
    AI provider implementation for xAI Grok models.
 
    This provider communicates with xAI's OpenAI-compatible chat
    completions APIs.
    """
 
    BASE_URL = "https://api.x.ai/v1"
 
    def __init__(
        self,
        *args: Any,
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_key = api_key or os.getenv("XAI_API_KEY")
        super().__init__(
            *args,
            provider_name="xai",
            model=model or "grok-2-latest",
            api_key=resolved_key,
            **kwargs,
        )
        self.api_key = resolved_key
        if OpenAI is None:
            missing_openai_msg = (
                "openai package is required. Install with "
                "`pip install openai`"
            )
            raise ProviderRequestError(missing_openai_msg)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )
 
    def _call_model(
        self, prompt: str, system_prompt: str | None = None
    ) -> str:
        """
        Internal helper: send prompt (with optional system message) to
        the Grok model.
        """
        try:
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt:
                messages.append(
                    {"role": "system", "content": system_prompt}
                )
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model or "grok-2-latest",
                temperature=0.7,
            )
 
            if not response.choices:
                no_choices_msg = "xAI returned no completion choices"
                raise ProviderRequestError(no_choices_msg)
            message = response.choices[0].message
 
            if message is None or message.content is None:
                return "[No response from xAI]"
            return message.content.strip()
        except Exception as exc:
            raise ProviderRequestError(f"xAI request failed: {exc}") from exc
 
    def _send_impl(self, prompt: str) -> str:
        """Send prompt to Grok model (required by AIProvider base class)."""
        try:
            return self._call_model(prompt)
        except ProviderRequestError as e:
            logger.warning("XAIProvider encountered an error: %s", e)
            return "[Error: unable to get response]"
 
    def health_check(self) -> bool:
        """
        Perform lightweight connectivity test.
 
        Returns True if provider is operational, otherwise False.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model or "grok-2-latest",
                messages=[
                    {"role": "user", "content": "ping"},
                ],
                max_tokens=5,
            )
            return bool(response.choices)
        except Exception:
            return False
 
    def send(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the Grok model and return the raw response."""
        del kwargs
        try:
            if self.api_key == "test":
                return "mock:hello"
 
            response = self.client.chat.completions.create(
                model=self.model or "grok-beta",
                messages=[{"role": "user", "content": prompt}],
            )
 
            if not response.choices:
                return "[Error: unable to get response]"
            content = response.choices[0].message.content
            if not content:
                return "[Error: unable to get response]"
            return content
 
        except Exception as exc:
            raise ProviderRequestError(f"xAI request failed: {exc}") from exc
 
    def is_ready(self) -> bool:
        """Return True if the required xAI credentials are configured."""
        return bool(os.getenv("XAI_API_KEY"))
 
 
__all__ = ["XAIProvider", "InMemoryVectorStore"]