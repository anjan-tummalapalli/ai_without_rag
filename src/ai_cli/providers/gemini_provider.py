from __future__ import annotations

import warnings
from typing import Any

from ai_cli.core.exceptions import ProviderRequestError


class _GenaiShim:
    """Shim that raises ProviderRequestError when no Google SDK is found."""

    def configure(self, *_args: Any, **_kwargs: Any) -> None:
        raise ProviderRequestError(
            "Google Generative AI SDK is not installed; "
            "install 'google-generativeai' or 'google-genai'."
        )

    class GenerativeModel:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ProviderRequestError(
                "Google Generative AI SDK is not installed; "
                "cannot create model."
            )

    class Client:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ProviderRequestError(
                "Google Generative AI SDK is not installed; "
                "cannot create client."
            )

        class models:
            @staticmethod
            def generate_content(*_args: Any, **_kwargs: Any) -> None:
                raise ProviderRequestError(
                    "Google Generative AI SDK is not installed; "
                    "cannot generate content."
                )


genai: Any

try:
    import google.generativeai as genai
    _GENAI_LEGACY = True
except ImportError:
    try:
        from google import genai
        _GENAI_LEGACY = False
    except ImportError:
        genai = _GenaiShim()
        _GENAI_LEGACY = False

warnings.filterwarnings("ignore", category=FutureWarning)