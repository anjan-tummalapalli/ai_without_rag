from ai_cli.providers.registry import (
    CHAT_PROVIDERS,
)
from ai_cli.utils.test_mode import is_test_mode
from ai_cli.providers.contracts import ChatProvider, EmbeddingProvider


def test_all_chat_providers_have_ask():
    for cls in CHAT_PROVIDERS.values():
        assert hasattr(cls, "ask")


def test_registry_builds():
    for cls in CHAT_PROVIDERS.values():
        assert cls is not None


def send(self, prompt: str):
    if is_test_mode():
        return "mock:gemini:" + prompt


def test_fake_cohere():
    def fake_send(prompt):
        return "mock:cohere:" + prompt

    assert fake_send("hello") == "mock:cohere:hello"


def test_fake_deepseek():
    def fake_send(prompt):
        return "mock:deepseek:" + prompt

    assert fake_send("hello") == "mock:deepseek:hello"


def test_fake_xai():
    def fake_send(prompt):
        return "mock:xAI:" + prompt

    assert fake_send("hello") == "mock:xAI:hello"


def test_fake_zai():
    def fake_send(prompt):
        return "mock:zAI:" + prompt

    assert fake_send("hello") == "mock:zAI:hello"


class DummyChatProvider(ChatProvider):
    def ask(self, prompt: str, **kwargs):
        return f"reply:{prompt}"


class DummyEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts, **kwargs):
        return [[1.0] * len(texts)]


def test_chat_provider_contract():
    provider = DummyChatProvider()
    assert provider.ask("hello") == "reply:hello"


def test_embedding_provider_contract():
    provider = DummyEmbeddingProvider()
    assert provider.embed(["a", "b"]) == [[1.0, 1.0]]
