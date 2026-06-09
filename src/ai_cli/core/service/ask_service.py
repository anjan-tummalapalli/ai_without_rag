import os
from ai_cli.providers.registry import get_chat_provider
from ai_cli.providers.base import ProviderConfig


def ask(
    prompt: str,
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
):

    if api_key is None:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")

    config = ProviderConfig(
        model=model,
        api_key=api_key,
        timeout=timeout,
    )

    provider_obj = get_chat_provider(provider, config=config)
    return provider_obj.chat(prompt)
