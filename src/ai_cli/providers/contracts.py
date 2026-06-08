from abc import ABC, abstractmethod
from typing import List, Any


class ChatProvider(ABC):
    def __init__(self, *, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def send(self, prompt: str, **kwargs) -> str:
        pass


class EmbeddingProvider(ABC):
    def __init__(self, *, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass


class RAGProvider(ABC):
    def __init__(self, chat: ChatProvider, embeddings: EmbeddingProvider):
        self.chat = chat
        self.embeddings = embeddings

    @abstractmethod
    def ask(self, query: str, docs: List[str]) -> str:
        pass