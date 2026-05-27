"""
Gemini provider implementation for ai_cli.

This module provides integration with Google's Gemini models
using the official google-generativeai SDK.

Environment Variables
---------------------
GEMINI_API_KEY
    API key used to authenticate with Google Gemini API.

Example
-------
export GEMINI_API_KEY="your_api_key"

Usage
-----
provider = GeminiProvider(
    model="gemini-1.5-flash"
)

response = provider.send("Explain Kubernetes operators")
print(response)
"""

from __future__ import annotations

import os
from typing import Optional

import google.generativeai as genai

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider


class GeminiProvider(AIProvider):
    """
    AI provider implementation for Google Gemini models.

    This provider uses the official Google Generative AI SDK
    to send prompts and receive responses from Gemini models.

    Supported Models
    ----------------
    - gemini-1.5-flash
    - gemini-1.5-pro
    - gemini-2.0-flash
    - Other Gemini-compatible models

    Attributes
    ----------
    api_key : str
        Gemini API key loaded from environment variable.

    model : str
        Model name used for inference.

    client : genai.GenerativeModel
        Configured Gemini model client instance.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initialize Gemini provider.

        Parameters
        ----------
        model : str | None, optional
            Gemini model name to use.
            Defaults to 'gemini-1.5-flash'.

        api_key : str | None, optional
            Gemini API key.
            If not provided, reads from GEMINI_API_KEY
            environment variable.

        *args
            Additional positional arguments passed to AIProvider.

        **kwargs
            Additional keyword arguments passed to AIProvider.

        Raises
        ------
        ProviderRequestError
            If API key is missing or invalid.
        """

        super().__init__(
            model=model or "gemini-1.5-flash",
            api_key=api_key,
            *args,
            **kwargs,
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ProviderRequestError(
                "GEMINI_API_KEY environment variable is not set"
            )

        # Configure Gemini SDK
        genai.configure(api_key=self.api_key)
        # Create model client
        self.client = genai.GenerativeModel(self.model)

    def _send_impl(self, prompt: str) -> str:
        """
        Send prompt to Gemini model and return generated response.
        This method performs the actual API request to Gemini.

        Parameters
        ----------
        prompt : str
            User prompt or query text to send to Gemini.

        Returns
        -------
        str
            Generated text response from Gemini model.

        Raises
        ------
        ProviderRequestError
            If Gemini API request fails or response is invalid.

        Examples
        --------
        >>> provider._send_impl("Explain Kubernetes")
        'Kubernetes is a container orchestration platform...'
        """
        try:
            response = self.client.generate_content(prompt)
            if not response:
                raise ProviderRequestError(
                    "Gemini returned empty response"
                )
            if not hasattr(response, "text"):
                raise ProviderRequestError(
                    "Gemini response missing text field"
                )
            return response.text.strip()

        except Exception as exc:
            raise ProviderRequestError(
                f"Gemini request failed: {exc}"
            ) from exc

    def health_check(self) -> bool:
        """
        Perform lightweight connectivity test against Gemini API.

        Returns
        -------
        bool
            True if provider is operational, otherwise False.

        Notes
        -----
        This method sends a minimal prompt to verify:
        - API key validity
        - Network connectivity
        - Gemini service availability
        """
        try:
            response = self.client.generate_content("ping")
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
        return "gemini"