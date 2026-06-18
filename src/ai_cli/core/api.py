from ai_cli.providers.registry import build_provider


def ask(prompt, provider, model=None, _provider=None, **kwargs):
    ai_provider = _provider or build_provider(
        provider_name=provider,
        model=model,
        **kwargs,
    )
    return ai_provider.send(prompt)