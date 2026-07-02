from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ChatProvider(ABC):
    @abstractmethod
    def ask(self, prompt: str, **kwargs: Any) -> str:
        pass

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        pass