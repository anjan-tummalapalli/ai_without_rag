"""
Perplexity provider implementation for ai_cli.

This module integrates Perplexity AI models into the ai_cli
provider framework using Perplexity's OpenAI-compatible API.

Environment Variables
---------------------
PERPLEXITY_API_KEY
    API key used to authenticate with Perplexity API.

Example
-------
export PERPLEXITY_API_KEY="your_api_key"

Usage
-----
provider = PerplexityProvider(
    model="sonar-pro"
)

response = provider.send("Explain Kubernetes operators")
print(response)
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider


class PerplexityProvider(AIProvider):
    """
    AI provider implementation for Perplexity AI.

    This provider uses the OpenAI-compatible Perplexity API
    endpoint to send prompts and receive completions.

    Supported Models
    ----------------
    - sonar
    - sonar-pro
    - sonar-reasoning
    - sonar-deep-research

    Attributes
    ----------
    api_key : str
        Perplexity API key.

    model : str
        Model used for inference.

    client : OpenAI
        OpenAI-compatible client configured for Perplexity.
    """

    BASE_URL = "https://api.perplexity.ai"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize Perplexity provider.

        Parameters
        ----------
        model : str | None, optional
            Perplexity model name.
            Defaults to 'sonar-pro'.

        api_key : str | None, optional
            Perplexity API key.
            If not provided, reads from
            PERPLEXITY_API_KEY environment variable.

        *args
            Additional positional arguments passed to AIProvider.

        **kwargs
            Additional keyword arguments passed to AIProvider.

        Raises
        ------
        ProviderRequestError
            If API key is missing.
        """
        super().__init__(
            model=model or "sonar-pro",
            api_key=api_key,
            *args,
            **kwargs,
        )
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")

        if not self.api_key:
            raise ProviderRequestError(
                "PERPLEXITY_API_KEY environment variable is not set"
            )
        # Create OpenAI-compatible client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    def _send_impl(self, prompt: str) -> str:
        """
        Send prompt to Perplexity model.

        Parameters
        ----------
        prompt : str
            User prompt to send to Perplexity.

        Returns
        -------
        str
            Generated response text.

        Raises
        ------
        ProviderRequestError
            If API request fails or response is invalid.

        Examples
        --------
        >>> provider._send_impl("What is Kubernetes?")
        'Kubernetes is an open-source container orchestration platform...'
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            if not response.choices:
                raise ProviderRequestError(
                    "Perplexity returned no choices"
                )
            message = response.choices[0].message
            if not message or not message.content:
                raise ProviderRequestError(
                    "Perplexity returned empty content"
                )
            return message.content.strip()
        except Exception as exc:
            raise ProviderRequestError(
                f"Perplexity request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Perform lightweight provider health check.

        Returns
        -------
        bool
            True if provider is operational,
            otherwise False.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": "ping",
                    }
                ],
            )
            return bool(response.choices)
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        """
        Return provider identifier.

        Returns
        -------
        str
            Provider name.
        """
        return "perplexity"