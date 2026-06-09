from ai_cli.providers.bootstrap import init_providers
from ai_cli.providers.registry import build_provider

def ask(provider: str, prompt: str, **kwargs):
    init_providers()  # safe, idempotent
    ai_provider = build_provider(provider)
    return ai_provider.ask(prompt, **kwargs)
