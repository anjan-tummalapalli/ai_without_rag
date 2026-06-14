
from ai_cli.providers.contracts import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("embed() not implemented")
