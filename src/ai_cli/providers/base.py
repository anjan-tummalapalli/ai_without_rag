from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# =========================================================
# Core metadata model (used by registry/tests)
# =========================================================

@dataclass
class ProviderMetadata:
    name: str


# =========================================================
# Base provider contract
# =========================================================

class BaseProvider:
    metadata: ProviderMetadata
    def __init__(self, **kwargs):
        self.api_key = kwargs.get("api_key")
        self.model = kwargs.get("model")

    # -------------------------
    # Chat
    # -------------------------
    def embed(self, *args, **kwargs):
        """
        Optional embedding interface. Providers that support embeddings should
        override this method. Default implementation raises NotImplementedError
        so the attribute exists for provider contract checks/tests.
        """
        raise NotImplementedError("embed() not implemented by this provider")

    def send(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    def ask(self, prompt: str, **kwargs) -> str:
        return self.send(prompt, **kwargs)

    # -------------------------

    # -------------------------
    def upsert_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int = 5):
        raise NotImplementedError
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        if not text:
            return []

        step = max(chunk_size - overlap, 1)
        chunks = []
        for start in range(0, len(text), step):
            chunks.append(text[start:start + chunk_size])
            if start + chunk_size >= len(text):
                break
        return chunks

# =========================================================
# Legacy alias compatibility (THIS FIXES YOUR ERRORS)
# =========================================================

# Some tests expect AIProvider instead of BaseProvider
AIProvider = BaseProvider

# Some tests expect EchoProvider in base (yes, weird but required by your tests)
class EchoProvider(BaseProvider):
    """
    Simple deterministic provider used in tests.
    """
    provider_name = "echo"

    def __init__(self, **kwargs):
        super().__init__(model="echo", **kwargs)
        self.provider_name = "echo"

    def send(self, prompt: str, **kwargs) -> str:
        return f"(echo) {prompt}"

    def upsert_documents(self, texts, metadatas=None):
        return None

    def retrieve(self, query: str, top_k: int = 5):
        return []

    def ask(self, prompt: str, **kwargs) -> str:
        return self.send(prompt, **kwargs)

__all__ = [
    "BaseProvider",
    "AIProvider",
    "ProviderMetadata",
    "EchoProvider",
]
