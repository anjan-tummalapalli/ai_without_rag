from __future__ import annotations
 
from dataclasses import dataclass
from typing import Any
 
 
@dataclass
class ProviderMetadata:
    """Metadata describing a provider."""
 
    name: str
    supports_rag: bool = False
 
 
class BaseProvider:
    """Base class for all AI providers."""
 
    metadata: ProviderMetadata
 
    def __init__(self, **kwargs: Any) -> None:
        self.api_key = kwargs.get("api_key")
        self.model = kwargs.get("model")
 
    def send(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError
 
    def ask(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return self.send(prompt, **kwargs)
 
    def chat(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Default chat implementation; delegates to ask().
 
        Providers with their own chat-specific behavior (e.g. distinct
        error wrapping) should override this.
        """
        return self.ask(prompt, **kwargs)
 
 
# Backwards compatibility
AIProvider = BaseProvider
 
 
class EchoProvider(BaseProvider):
    """Simple deterministic provider used in tests."""
 
    provider_name = "echo"
 
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model="echo", **kwargs)
 
    def send(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return f"(echo) {prompt}"
 
 
__all__ = [
    "BaseProvider",
    "AIProvider",
    "ProviderMetadata",
    "EchoProvider",
]