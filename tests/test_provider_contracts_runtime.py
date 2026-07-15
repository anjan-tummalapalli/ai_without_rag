from ai_cli.providers.contracts import ChatProvider, EmbeddingProvider


class ChatImpl(ChatProvider):
    def ask(self, prompt: str, **kwargs):
        return f"reply:{prompt}"


class EmbedImpl(EmbeddingProvider):
    def embed(self, texts, **kwargs):
        return [[float(len(t))] for t in texts]


def test_chat_provider_runtime():
    p = ChatImpl()
    assert p.ask("hello") == "reply:hello"


def test_embedding_provider_runtime():
    p = EmbedImpl()
    assert p.embed(["a", "bb"]) == [[1.0], [2.0]]
