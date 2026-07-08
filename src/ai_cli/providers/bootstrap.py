from ai_cli.providers.loader import load_all_providers
from ai_cli.providers.registry import (
    register_chat_provider,
    register_provider,
)

_initialized = False  # pylint: disable=invalid-name


def init_providers() -> None:
    """Load and register all providers once per process."""
    global _initialized  # pylint: disable=global-statement
    if _initialized:
        return
    for name, cls in load_all_providers().items():
        register_provider(name, cls)
        register_chat_provider(name, cls)
    _initialized = True
