"""
Cohere provider implementation for ai_cli.

This module integrates Cohere large language models into the
ai_cli provider framework using the official Cohere Python SDK.

Environment Variables
---------------------
COHERE_API_KEY
    API key used to authenticate with Cohere API.

Example
-------
export COHERE_API_KEY="your_api_key"

Usage
-----
provider = CohereProvider(
    model="command-r"
)

response = provider.send("Explain Kubernetes operators")
print(response)
"""

from __future__ import annotations

import os
from typing import Optional

import cohere

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider


class CohereProvider(AIProvider):
    """
    AI provider implementation for Cohere models.

    This provider communicates with Cohere's chat API
    using the official Cohere SDK.

    Supported Models
    ----------------
    - command-r
    - command-r-plus
    - command
    - Other Cohere-compatible models

    Attributes
    ----------
    api_key : str
        Cohere API key.

    model : str
        Cohere model name.

    client : cohere.Client
        Configured Cohere SDK client instance.
    """
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize Cohere provider.

        Parameters
        ----------
        model : str | None, optional
            Cohere model name.
            Defaults to 'command-r'.

        api_key : str | None, optional
            Cohere API key.
            If not provided, reads from
            COHERE_API_KEY environment variable.

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
            model=model or "command-r",
            api_key=api_key,
            *args,
            **kwargs,
        )

        self.api_key = api_key or os.getenv("COHERE_API_KEY")

        if not self.api_key:
            raise ProviderRequestError(
                "COHERE_API_KEY environment variable is not set"
            )
        # Initialize Cohere client
        self.client = cohere.Client(self.api_key)

    def _send_impl(self, prompt: str) -> str:
        """
        Send prompt to Cohere model and return response.

        Parameters
        ----------
        prompt : str
            User prompt or query text.

        Returns
        -------
        str
            Generated response text from Cohere.

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
            response = self.client.chat(
                model=self.model,
                message=prompt,
            )
            if not response:
                raise ProviderRequestError(
                    "Cohere returned empty response"
                )
            if not hasattr(response, "text"):
                raise ProviderRequestError(
                    "Cohere response missing text field"
                )

            if not response.text:
                raise ProviderRequestError(
                    "Cohere returned empty text response"
                )
            return response.text.strip()

        except Exception as exc:
            raise ProviderRequestError(
                f"Cohere request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Perform lightweight Cohere connectivity test.

        Returns
        -------
        bool
            True if provider is operational,
            otherwise False.

        Notes
        -----
        Sends a minimal request to verify:
        - API key validity
        - Network connectivity
        - Cohere service availability
        """
        try:
            response = self.client.chat(
                model=self.model,
                message="ping",
            )
            return bool(response and response.text)
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
        return "cohere"