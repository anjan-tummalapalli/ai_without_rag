import os

from ai_cli.providers.registry import build_provider


def embed(
    texts: list[str],
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
):

    if api_key is None:
        api_key = os.getenv(f"{provider.upper()}_API_KEY")

    provider_obj = build_provider(
        provider,
        model=model,
        api_key=api_key,
    )

    if not hasattr(provider_obj, "embed"):
        raise TypeError(f"{provider} does not support embeddings")

    return provider_obj.embed(texts)
