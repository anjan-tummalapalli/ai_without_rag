from abc import ABC, abstractmethod
from typing import Any, Optional

class ChatProvider(ABC):

    @abstractmethod
    def chat(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> str:


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(
        self,
        texts: list[str],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> list[list[float]]:

    

class RAGProvider(ChatProvider, EmbeddingProvider):
    """
    Optional extension interface.
    """
    def add_documents(self, docs: list[str]) -> None:
