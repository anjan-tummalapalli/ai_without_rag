import os
from ai_cli.providers.registry import build_provider

def ask(**kwargs):
    provider = kwargs.get("provider", "auto")
    # -----------------------------
    # FIX: ensure API key exists
    # -----------------------------
    if "api_key" not in kwargs or kwargs["api_key"] is None:
        kwargs["api_key"] = os.getenv(f"{provider.upper()}_API_KEY")
    ai_provider = build_provider(provider, **kwargs)
    return ai_provider.send(kwargs["prompt"])
