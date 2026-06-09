from abc import ABC, abstractmethod
from typing import List

class ChatProvider(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        pass

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass

class RAGProvider(ChatProvider, EmbeddingProvider):
    @abstractmethod
    def add_documents(self, documents):
        pass
