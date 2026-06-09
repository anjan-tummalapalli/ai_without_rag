from typing import Type

from ai_cli.providers.registry import (
    register_provider,
    register_chat_provider,
    register_embedding_provider,
)


def provider(name: str):
    def wrapper(cls: Type):
        register_provider(name, cls)
        return cls
    return wrapper


def chat_provider(name: str):
    def wrapper(cls: Type):
        register_chat_provider(name, cls)
        return cls
    return wrapper


def embedding_provider(name: str):
    def wrapper(cls: Type):
        register_embedding_provider(name, cls)
        return cls
    return wrapper
