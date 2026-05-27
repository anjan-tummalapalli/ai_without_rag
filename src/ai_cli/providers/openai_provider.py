"""
OpenAI ChatGPT provider implementation for ai_cli.

This module integrates OpenAI ChatGPT models into the ai_cli
provider framework using the official OpenAI Python SDK.

Environment Variables
---------------------
OPENAI_API_KEY
    API key used to authenticate with OpenAI API.

Example
-------
export OPENAI_API_KEY="your_api_key"

Usage
-----
provider = OpenAIProvider(
    model="gpt-5.5"
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


class OpenAIProvider(AIProvider):
    """
    AI provider implementation for OpenAI ChatGPT models.

    This provider communicates with OpenAI's Chat Completions API.

    Supported Models
    ----------------
    - gpt-5.5
    - gpt-4o
    - gpt-4.1
    - gpt-4-turbo
    - gpt-3.5-turbo

    Attributes
    ----------
    api_key : str
        OpenAI API key.

    model : str
        OpenAI model name.

    client : OpenAI
        OpenAI SDK client instance.
    """

    BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize OpenAI provider.

        Parameters
        ----------
        model : str | None, optional
            OpenAI model name.
            Defaults to 'gpt-5.5'.

        api_key : str | None, optional
            OpenAI API key.
            If not provided, reads from
            OPENAI_API_KEY environment variable.

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
            model=model or "gpt-5.5",
            api_key=api_key,
            *args,
            **kwargs,
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ProviderRequestError(
                "OPENAI_API_KEY environment variable is not set"
            )

        # Create OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL,
        )

    def _send_impl(self, prompt: str) -> str:
        """
        Send prompt to OpenAI ChatGPT model.

        Parameters
        ----------
        prompt : str
            User prompt or query.

        Returns
        -------
        str
            Generated response text.

        Raises
        ------
        ProviderRequestError
            If request fails or response is invalid.

        Examples
        --------
        >>> provider._send_impl("What is Kubernetes?")
        'Kubernetes is a container orchestration platform...'
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
                    "OpenAI returned no completion choices"
                )

            message = response.choices[0].message

            if not message or not message.content:
                raise ProviderRequestError(
                    "OpenAI returned empty response content"
                )

            return message.content.strip()

        except Exception as exc:
            raise ProviderRequestError(
                f"OpenAI request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Perform lightweight OpenAI connectivity test.

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
        return "openai"