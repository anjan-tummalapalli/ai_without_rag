"""
Provider bootstrap loader (deterministic, no side effects).
"""

from ai_cli.providers.base import EchoProvider
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider


def load_all_providers():
    return {
        "echo": EchoProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "cohere": CohereProvider,
        "perplexity": PerplexityProvider,
        "xai": XAIProvider,
        "zai": ZAIProvider,
    }