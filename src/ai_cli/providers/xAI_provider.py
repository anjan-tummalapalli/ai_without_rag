"""
xAI Grok provider implementation for ai_cli.

This module integrates xAI Grok models into the ai_cli
provider framework using xAI's OpenAI-compatible API.

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
print(response)
"""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider


class XAIProvider(AIProvider):
    """
    AI provider implementation for xAI Grok models.

    This provider communicates with xAI's OpenAI-compatible
    chat completions API.

    Supported Models
    ----------------
    - grok-2-latest
    - grok-2-mini
    - grok-beta
    - Other Grok-compatible models

    Attributes
    ----------
    api_key : str
        xAI API key.

    model : str
        Grok model name.

    client : OpenAI
        OpenAI-compatible API client configured for xAI.
    """

    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize xAI provider.

        Parameters
        ----------
        model : str | None, optional
            Grok model name.
            Defaults to 'grok-2-latest'.

        api_key : str | None, optional
            xAI API key.
            If not provided, reads from
            XAI_API_KEY environment variable.

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
            model=model or "grok-2-latest",
            api_key=api_key,
            *args,
            **kwargs,
        )
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        if not self.api_key:
            raise ProviderRequestError(
                "XAI_API_KEY environment variable is not set"
            )
        # Create OpenAI-compatible client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    def _send_impl(self, prompt: str) -> str:
        """
        Send prompt to Grok model and return generated response.

        Parameters
        ----------
        prompt : str
            User input prompt.

        Returns
        -------
        str
            Generated response text from Grok.

        Raises
        ------
        ProviderRequestError
            If request fails or response is invalid.

        Examples
        --------
        >>> provider._send_impl("Explain Kubernetes")
        'Kubernetes is an orchestration platform...'
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
                temperature=0.7,
            )

            if not response.choices:
                raise ProviderRequestError(
                    "xAI returned no completion choices"
                )
            message = response.choices[0].message

            if not message or not message.content:
                raise ProviderRequestError(
                    "xAI returned empty response content"
                )
            return message.content.strip()
        except Exception as exc:
            raise ProviderRequestError(
                f"xAI request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Perform lightweight connectivity test.

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
                max_tokens=5,
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
        return "xai"