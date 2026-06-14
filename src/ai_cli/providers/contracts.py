from __future__ import annotations

from abc import ABC, abstractmethod


class ChatProvider(ABC):
    @abstractmethod
    def ask(self, prompt: str, **kwargs) -> str:
        pass

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass

class RAGProvider(ChatProvider, EmbeddingProvider):
    @abstractmethod
    def add_documents(self, documents):
        pass
