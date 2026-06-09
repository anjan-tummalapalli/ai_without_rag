from ai_cli.providers.loader import load_all_providers
from ai_cli.providers.registry import (
    register_provider,
    register_chat_provider,
)

_initialized = False

def init_providers() -> None:
    global _initialized
    if _initialized:
        return
    providers = load_all_providers()
    for name, cls in providers.items():
        register_provider(name, cls)
        register_chat_provider(name, cls)
    _initialized = True
