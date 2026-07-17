from typing import Any
 
from ai_cli.providers.registry import build_provider, ensure_initialized
 
 
def ask(
    prompt: str,
    provider: str,
    model: str | None = None,
    _provider: Any = None,
    **kwargs: Any,
) -> str:
    ensure_initialized()
 
    ai_provider = _provider or build_provider(
        name=provider,
        model=model if model else "gpt-4o-mini",
        **kwargs,
    )
 
    if model is None:
        model = "gpt-4o-mini"
 
    result = ai_provider.send(prompt)
    return result