from collections.abc import Callable

from ai_cli.providers.registry import (
    register_chat_provider,
    register_provider,
)


def provider(name: str) -> Callable[[type], type]:
    def wrapper(cls: type) -> type:
        register_provider(name, cls)
        return cls

    return wrapper


def chat_provider(name: str) -> Callable[[type], type]:
    def wrapper(cls: type) -> type:
        register_chat_provider(name, cls)
        return cls

    return wrapper
