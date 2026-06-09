from ai_cli.providers.registry import (
    CHAT_PROVIDERS,
    EMBEDDING_PROVIDERS,
)

def test_all_chat_providers_have_ask():
    for cls in CHAT_PROVIDERS.values():
        assert hasattr(cls, "ask")

def test_all_embedding_providers_have_embed():
    for cls in EMBEDDING_PROVIDERS.values():
        assert hasattr(cls, "embed")

def test_registry_builds():
    for cls in CHAT_PROVIDERS.values():
        assert cls is not None
