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