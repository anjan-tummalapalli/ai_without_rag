from __future__ import annotations

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore

from ai_cli.config.rag_config import EMBEDDING_MODEL


class EmbeddingGenerator:
    """
    Generates embeddings using SentenceTransformers.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        if SentenceTransformer is None:
            raise ImportError(
                "The 'sentence_transformers' package is required but not installed. "
                "Install it with: pip install sentence-transformers"
            )
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