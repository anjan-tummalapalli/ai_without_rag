from __future__ import annotations

from ai_cli.config.rag_config import TOP_K
from ai_cli.rag.embeddings import EmbeddingGenerator
from ai_cli.rag.vector_store import VectorStore


class Retriever:
    """
    Retrieves semantically similar chunks.
    """

    def __init__(self) -> None:
        self.embeddings = EmbeddingGenerator()
        self.store = VectorStore()
        self.store.load()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
    ):
        """
        Retrieve top matching chunks.
        """

        query_embedding = self.embeddings.embed_text(query)

        return self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )