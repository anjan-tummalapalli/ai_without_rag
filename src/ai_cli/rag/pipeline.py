from __future__ import annotations

from ai_cli.rag.chunker import SemanticChunker
from ai_cli.rag.document_loader import DocumentLoader
from ai_cli.rag.embeddings import EmbeddingGenerator
from ai_cli.rag.retriever import Retriever
from ai_cli.rag.vector_store import VectorStore


class RAGPipeline:
    """
    Full RAG orchestration pipeline.
    """

    def __init__(self) -> None:
        self.loader = DocumentLoader()
        self.chunker = SemanticChunker()
        self.embeddings = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.vector_store.load()

    def index_document(
        self,
        file_path: str,
    ) -> None:
        """
        Index a single document.
        """

        document = self.loader.load(file_path)

        chunks = self.chunker.chunk_text(
            text=document.content,
            source=document.source,
        )

        embeddings = self.embeddings.embed_batch(
            [chunk.text for chunk in chunks]
        )

        self.vector_store.add_embeddings(
            embeddings,
            chunks,
        )

        self.vector_store.save()

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve relevant context.
        """

        retriever = Retriever()

        results = retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        contexts = []

        for result in results:
            chunk = result["chunk"]
            contexts.append(chunk.text)

        return "\n\n".join(contexts)