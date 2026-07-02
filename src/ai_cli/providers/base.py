from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# =========================================================
# Core metadata model (used by registry/tests)
# =========================================================

@dataclass
class ProviderMetadata:
    name: str
    supports_rag: bool = False


# =========================================================
# Base provider contract
# =========================================================

class BaseProvider:
    metadata: ProviderMetadata
    def __init__(self, **kwargs: Any) -> None:
        self.api_key = kwargs.get("api_key")
        self.model = kwargs.get("model")

    # -------------------------
    # Chat
    # -------------------------
    def send(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError
    
    def ask(self, prompt: str, **kwargs: Any) -> str:
        return self.send(prompt, **kwargs)

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="echo", **kwargs)
        self.provider_name = "echo"

    def send(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return f"(echo) {prompt}"

    def ask(self, prompt: str, **kwargs: Any) -> str:
        return self.send(prompt, **kwargs)

__all__ = [
    "BaseProvider",
    "AIProvider",
    "ProviderMetadata",
    "EchoProvider",
]
