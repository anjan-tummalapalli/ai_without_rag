
from ai_cli.providers.registry import (
    register_chat_provider,
    register_embedding_provider,
    register_provider,
)


def provider(name: str):
    def wrapper(cls: type):
        register_provider(name, cls)
        return cls
    return wrapper


def chat_provider(name: str):
    def wrapper(cls: type):
        register_chat_provider(name, cls)
        return cls
    return wrapper


def embedding_provider(name: str):
    def wrapper(cls: type):
        register_embedding_provider(name, cls)
        return cls
    return wrapper
