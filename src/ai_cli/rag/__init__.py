"""RAG utilities: chunking, embeddings, vector storage, and retrieval."""

from ai_cli.rag.chunker import SemanticChunker, chunk_text
from ai_cli.rag.embeddings import EmbeddingGenerator, EmbeddingsProvider
from ai_cli.rag.in_memory import InMemoryRAGPipeline, RAGPipeline
from ai_cli.rag.models import Chunk, Document, Embedding, RetrievalResult
from ai_cli.rag.retriever import Retriever
from ai_cli.rag.vector_store import InMemoryVectorStore, VectorStore

__all__ = [
    "Chunk",
    "Document",
    "Embedding",
    "EmbeddingGenerator",
    "EmbeddingsProvider",
    "InMemoryRAGPipeline",
    "InMemoryVectorStore",
    "RAGPipeline",
    "RetrievalResult",
    "Retriever",
    "SemanticChunker",
    "VectorStore",
    "chunk_text",
]
