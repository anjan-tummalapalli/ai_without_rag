"""

Provider bootstrap loader (deterministic, no side effects).

"""


def load_all_providers():
    from ai_cli.providers.auto_provider import AutoProvider
    from ai_cli.providers.cohere_provider import CohereProvider
    from ai_cli.providers.deepseek_provider import DeepSeekProvider
    from ai_cli.providers.echo_provider import EchoProvider
    from ai_cli.providers.gemini_provider import GeminiProvider
    from ai_cli.providers.openai_provider import OpenAIProvider
    from ai_cli.providers.perplexity_provider import PerplexityProvider
    from ai_cli.providers.xAI_provider import XAIProvider
    from ai_cli.providers.zAI_provider import ZAIProvider

    return {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "deepseek": DeepSeekProvider,
        "perplexity": PerplexityProvider,
        "xai": XAIProvider,
        "cohere": CohereProvider,
        "zai": ZAIProvider,
        "echo": EchoProvider,
        "auto": AutoProvider,
    }
