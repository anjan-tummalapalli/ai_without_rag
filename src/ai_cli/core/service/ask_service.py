import os

from ai_cli.providers.registry import get_chat_provider


def ask(
    prompt: str,
    provider: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
):

    if api_key is None:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")

    provider_obj = get_chat_provider(
                                     provider,
                                     model=model,
                                     api_key=api_key,
                                     timeout=timeout,
                                    )

    return provider_obj.chat(prompt)
