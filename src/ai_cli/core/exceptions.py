# /Users/anjan/Documents/New project/ai_chat/ai_cli/src/ai_cli/core/exceptions.py
from __future__ import annotations

class AIProviderError(Exception):
    """Base exception for AI provider errors."""
    pass


class PromptValidationError(AIProviderError):
    """Raised when prompt validation fails."""
    pass


class ProviderConfigurationError(AIProviderError):
    """Raised when provider configuration is invalid."""
    pass


class ProviderRequestError(AIProviderError):
    """Raised when provider request execution fails."""
    pass


class ResponseValidationError(AIProviderError):
    """Raised when AI response validation fails."""
    pass


# New RAG-specific exceptions
class ChunkingError(AIProviderError):
    """Raised when text chunking fails."""
    pass


class EmbeddingError(AIProviderError):
    """Raised when embedding generation fails."""
    pass


class VectorDBError(AIProviderError):
    """Raised for vector DB/storage related errors."""
    pass


class RetrievalError(AIProviderError):
    """Raised when retrieval from vector store fails."""
    pass