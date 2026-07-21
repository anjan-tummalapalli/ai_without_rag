from typing import Any
 
from ai_cli.providers.registry import build_provider, ensure_initialized
 
DEFAULT_MODEL = "gpt-4o-mini"
 
 
def ask(
    prompt: str,
    provider: str,
    model: str | None = None,
    _provider: Any = None,
    **kwargs: Any,
) -> str:
    """Send `prompt` to `provider` (optionally overriding `model`) and
    return the response text.
 
    `_provider` is a test-only injection seam: pass a pre-built provider
    instance to bypass `build_provider`.
    """
    ensure_initialized()
 
    ai_provider = (
        _provider
        if _provider is not None
        else build_provider(
            name=provider,
            model=model or DEFAULT_MODEL,
            **kwargs,
        )
    )
 
    result = ai_provider.send(prompt)
    return result