from ai_cli.providers.registry import registry

from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider


def init_providers():
    registry.register("openai", OpenAIProvider)
    registry.register("gemini", GeminiProvider)
    registry.register("cohere", CohereProvider)
    registry.register("deepseek", DeepSeekProvider)
    registry.register("perplexity", PerplexityProvider)
    registry.register("xai", XAIProvider)
    registry.register("zai", ZAIProvider)
