"""ai_cli.providers package

Provides a public API for provider classes without importing them eagerly.

The ``__all__`` list contains the names of the public provider classes.  Tests
Provides a list of public provider class names without importing them.
This avoids circular import issues during package initialization.
"""
from .base import AIProvider
from .cohere_provider import CohereProvider
from .deepseek_provider import DeepSeekProvider
from .echo_provider import EchoProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .perplexity_provider import PerplexityProvider
from .xAI_provider import InMemoryVectorStore, XAIProvider
from .zAI_provider import ZAIProvider

__all__ = [
    "AIProvider",
    "XAIProvider",
    "InMemoryVectorStore",
    "OpenAIProvider",
    "GeminiProvider",
    "CohereProvider",
    "DeepSeekProvider",
    "PerplexityProvider",
    "ZAIProvider",
    "EchoProvider",
]
