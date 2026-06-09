from typing import List
from ai_cli.providers.contracts import EmbeddingProvider

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("embed() not implemented")
