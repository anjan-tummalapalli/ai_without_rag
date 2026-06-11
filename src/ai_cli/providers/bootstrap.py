from ai_cli.providers.loader import load_all_providers
from ai_cli.providers.registry import (
    register_provider,
    register_chat_provider,
)

def init_providers():
    providers = load_all_providers()
    for name, cls in providers.items():
        register_provider(name, cls)
        register_chat_provider(name, cls)
