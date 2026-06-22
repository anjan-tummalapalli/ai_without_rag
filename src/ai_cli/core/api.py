from ai_cli.providers.registry import build_provider


def ask(prompt, provider, model=None, _provider=None, **kwargs):
    ai_provider = _provider or build_provider(
        provider,
        model=model,
        **kwargs,
    )
    response = ai_provider.send(prompt)

    provider_name = getattr(
        ai_provider,
        "name",
        provider,
    )

    model_name = getattr(
        ai_provider,
        "model",
        model or "default",
    )
    return f"result from {provider_name}-{model_name}: {response}"