from ai_cli.providers.loader import load_all_providers
from ai_cli.providers.registry import (
    register_provider,
    register_chat_provider,
)

_initialized = False


def init_providers() -> None:
    """Load and register all providers once per process."""
    global _initialized
    if _initialized:
        return
    for name, cls in load_all_providers().items():
        register_provider(name, cls)
        register_chat_provider(name, cls)
    _initialized = True
