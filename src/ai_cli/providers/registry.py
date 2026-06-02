from __future__ import annotations

from ai_cli.providers.base import EchoProvider
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider

PROVIDER_MAP = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "cohere": CohereProvider,
    "deepseek": DeepSeekProvider,
    "perplexity": PerplexityProvider,
    "xai": XAIProvider,
}