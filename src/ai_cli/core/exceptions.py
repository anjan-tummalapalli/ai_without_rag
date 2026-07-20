# /Users/anjan/Documents/New project/ai_chat/ai_cli/src/ai_cli/core/exceptions.py
from __future__ import annotations

import json
import traceback
from typing import Any


class AIProviderError(Exception):
    """Base exception for AI provider errors with optional structured metadata.

    Args:
        message: Human readable message.
        code: Optional machine-readable error code.
        retryable: Whether the operation that caused this error can be retried.
        details: Arbitrary extra data useful for debugging (e.g., request_id, provider).
        cause: Optional original exception to chain.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.retryable = retryable
        self.details = details or {}
        # preserve exception chaining
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"(code={self.code})")
        if self.retryable:
            parts.append("[retryable]")
        if self.details:
            parts.append(f"details={self.details}")
        # keep it short; fallback to default if too long
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the exception (useful for logging or returning structured errors)."""
        return {
            "message": self.message,
            "code": self.code,
            "retryable": self.retryable,
            "details": self.details,
            "cause": repr(self.__cause__)
            if getattr(self, "__cause__", None)
            else None,
        }

    def to_json(self) -> str:
        """JSON representation (safe for logging)."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_exception(
        cls, exc: BaseException, *, message: str | None = None, **kwargs: Any
    ) -> AIProviderError:
        """Wrap an existing exception into an AIProviderError while preserving context."""
        msg = message or str(exc) or exc.__class__.__name__
        return cls(msg, cause=exc, **kwargs)


# Generic validation / configuration errors
class PromptValidationError(AIProviderError):
    """Raised when prompt validation fails."""

    pass


class ProviderConfigurationError(AIProviderError):
    """Raised when provider configuration is invalid."""

    pass


class ProviderRequestError(AIProviderError):
    """Raised when provider request execution fails.

    Common details: status_code, request_id, provider_name, response_body
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider_name: str | None = None,
        request_id: str | None = None,
        response_body: Any | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if status_code is not None:
            details["status_code"] = status_code
        if provider_name is not None:
            details["provider_name"] = provider_name
        if request_id is not None:
            details["request_id"] = request_id
        if response_body is not None:
            details["response_body"] = response_body
        super().__init__(message, details=details, **kwargs)


class ResponseValidationError(AIProviderError):
    """Raised when AI response validation fails."""

    pass


# RAG-specific exceptions
class ChunkingError(AIProviderError):
    """Raised when text chunking fails.

    details example: {"text_length": 12345, "chunk_size": 1024, "chunk_index": 5}
    """

    def __init__(
        self, message: str, *, chunk_index: int | None = None, **kwargs: Any
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if chunk_index is not None:
            details["chunk_index"] = chunk_index
        super().__init__(message, details=details, **kwargs)


class EmbeddingError(AIProviderError):
    """Raised when embedding generation fails.

    details example: {"model": "text-embedding-3", "input_tokens": 512}
    """

    def __init__(
        self, message: str, *, model: str | None = None, **kwargs: Any
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if model:
            details["model"] = model
        super().__init__(message, details=details, **kwargs)


class VectorDBError(AIProviderError):
    """Raised for vector DB/storage related errors.

    details example: {"operation": "upsert", "db": "faiss", "index_name": "docs_v1"}
    """

    def __init__(
        self, message: str, *, operation: str | None = None, **kwargs: Any
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)


class RetrievalError(AIProviderError):
    """Raised when retrieval from vector store fails.

    details example: {"query": "...", "retrieved": 0}
    """

    def __init__(
        self,
        message: str,
        *,
        query: str | None = None,
        retrieved: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {}) or {}
        if query is not None:
            details["query"] = query
        if retrieved is not None:
            details["retrieved"] = retrieved
        super().__init__(message, details=details, **kwargs)


# Small utility for capturing stack traces for debugging/logging without raising.
def capture_exception_info(exc: BaseException) -> dict[str, Any]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
