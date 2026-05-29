SentenceTransformer(
    "all-MiniLM-L6-v2"
)

from __future__ import annotations
from sentence_transformers import SentenceTransformer
from ai_cli.config.rag_config import EMBEDDING_MODEL


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def embed_text(
        self,
        text: str,
    ):
        """
        Generate embedding for a single text.
        """

        return self.model.encode(text)

    def embed_batch(
        self,
        texts: list[str],
    ):
        """
        Generate embeddings for multiple texts.
        """

        return self.model.encode(texts)