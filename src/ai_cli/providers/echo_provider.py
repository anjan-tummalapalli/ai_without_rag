"""
Echo provider for ai_cli – a simple mock that returns the prompt unchanged.
Used for testing and as a fallback when no real LLM is configured.
"""

from __future__ import annotations

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider


class EchoProvider(AIProvider):
    """A provider that simply echoes back the supplied prompt.

    This is useful for debugging, unit tests, or situations where a real
    language‑model API key is unavailable.
    """

    def __init__(self, model: str | None = "echo", api_key: str | None = None, *args, **kwargs):
        # No external service is required, but we keep the signature compatible.
        super().__init__(
                         *args,
                         provider_name="perplexity",
                         model=model or "sonar-pro",
                         api_key=api_key,
                         **kwargs,
                        )
        self.provider_name = "echo"

    def _send_impl(self, prompt: str) -> str:
        # The echo provider does not perform any network request. It just returns the prompt.
        if prompt is None:
            raise ProviderRequestError("Prompt cannot be None for EchoProvider")
        return prompt
