"""Base AIProvider class and EchoProvider for ai_cli."""
from __future__ import annotations
 
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any
 
from ai_cli.core.exceptions import (
    PromptValidationError,
    ProviderRequestError,
)
from ai_cli.core.resilience import RetryEngine
from ai_cli.telemetry.monitoring import ModelQualityMetrics
from ai_cli.utils.validation import HallucinationDetector, ResponseValidator
 
logger = logging.getLogger("ai_gateway")
 
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_PROMPT_LENGTH = 10_000
 
 
# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class ProviderMetadata:
    """Immutable metadata describing a provider's capabilities and cost."""
 
    name: str
    default_model: str
    supported_models: list[str]
    supports_streaming: bool
    supports_tools: bool
    supports_vision: bool
    max_context: int
    cost_per_1k_tokens: float
    avg_latency_ms: int
    supports_rag: bool = False
 
 
# pylint: disable=too-many-instance-attributes
class AIProvider:
    """Abstract base class for all AI providers.
 
    Subclasses must implement ``_send_impl(prompt)`` to perform the
    actual network call.  The public surface (``send``, ``chat``, ``ask``)
    handles validation, retries, metrics, and response coercion.
    """
 
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        provider_name: str | None = None,
        model: str | None = None,
        provider_meta: ProviderMetadata | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        embedding_client: Any | None = None,
        vector_db: Any | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        **kwargs,
    ) -> None:
        # Resolve provider_name: prefer explicit arg, fall back to class
        # attribute set by a subclass, then default to "unknown".
        name_val: str = provider_name or getattr(
            self, "provider_name", None
        ) or "unknown"
 
        # Guard against frozen/slotted subclasses that disallow assignment.
        try:
            self.provider_name: str = name_val
        except AttributeError:
            pass
 
        self.provider_meta: ProviderMetadata = (
            provider_meta
            or ProviderMetadata(
                name=name_val,
                default_model=model or "unknown",
                supported_models=[],
                supports_streaming=False,
                supports_tools=False,
                supports_vision=False,
                max_context=0,
                cost_per_1k_tokens=0.0,
                avg_latency_ms=0,
                supports_rag=False,
            )
        )
 
        self.timeout: int = timeout
        self.model: str = model or self.provider_meta.default_model
        self.api_key: str | None = kwargs.get("api_key")
 
        self.trace_id: str = str(uuid.uuid4())
 
        self.retry_engine = RetryEngine()
        self.response_validator = ResponseValidator()
        self.hallucination_detector = HallucinationDetector()
        self.metrics = ModelQualityMetrics(
            provider=name_val,
            model=self.model,
        )
 
        # RAG
        self.embedding_client = embedding_client
        self.vector_db = vector_db
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
 
    # -------------------------
    # Prompt validation
    # -------------------------
    def validate_prompt(self, prompt: str) -> str:
        """Validate and sanitize *prompt*.
 
        Raises PromptValidationError on any failure.
 
        Args:
            prompt: Raw user prompt string.
 
        Returns:
            Corrected, non-empty prompt within the allowed length limit.
 
        Raises:
            PromptValidationError: If the prompt is invalid or too long.
        """
        if not isinstance(prompt, str):
            raise PromptValidationError("prompt must be string")
 
        if "\x00" in prompt:
            raise PromptValidationError("prompt contains NUL byte")
 
        # Deferred import to avoid circular dependency at module load time.
        from ai_cli.core.prompt_corrector import (  # pylint: disable=C0415
            prompt_corrector,
        )
 
        corrected = prompt_corrector(prompt)
 
        if not corrected:
            raise PromptValidationError("prompt is empty")
 
        if len(corrected) > DEFAULT_MAX_PROMPT_LENGTH:
            raise PromptValidationError("prompt too long")
 
        return corrected
 
    # -------------------------
    # Abstract transport
    # -------------------------
    def _send_impl(self, prompt: str) -> str:
        """Perform the provider-specific network call.
 
        Args:
            prompt: Validated prompt string.
 
        Returns:
            Raw model response string.
 
        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError()
 
    # -------------------------
    # Response coercion
    # -------------------------
    def _coerce_response_to_str(self, response: Any) -> str:
        """Coerce any provider response value to a plain string.
 
        Args:
            response: Raw value returned by the provider SDK.
 
        Returns:
            String representation of the response.
 
        Raises:
            ProviderRequestError: If the response is None or un-coercible.
        """
        if response is None:
            raise ProviderRequestError("empty response")
 
        if isinstance(response, str):
            return response
 
        if isinstance(response, (dict, list, tuple, int, float, bool)):
            return json.dumps(response, ensure_ascii=False)
 
        try:
            return str(response)
        except Exception as exc:
            raise ProviderRequestError(
                f"Unsupported response type: {type(response)}"
            ) from exc
 
    # -------------------------
    # Main send
    # -------------------------
    def send(self, prompt: str) -> str:
        """Validate, send, and return the model response.
 
        Args:
            prompt: Raw user prompt.
 
        Returns:
            Stripped model response string.
 
        Raises:
            ProviderRequestError: On any transport or validation failure.
        """
        validated = self.validate_prompt(prompt)
 
        self.metrics.requests += 1
        start = time.monotonic()
 
        try:
            raw = self.retry_engine.execute(
                lambda: self._send_impl(validated)
            )
            result = self._coerce_response_to_str(raw)
            self.metrics.total_latency_seconds += time.monotonic() - start
            self.response_validator.validate(result)
            return result.strip()
        except Exception as exc:
            self.metrics.failures += 1
            raise ProviderRequestError(str(exc)) from exc
 
    def chat(self, prompt: str, **_kwargs: Any) -> str:
        """Standard chat interface; delegates to send().
 
        Args:
            prompt: User prompt.
            **_kwargs: Provider-specific options (for API compatibility).
 
        Returns:
            Model response string.
        """
        return self.send(prompt)
 
    def ask(self, prompt: str, **kwargs: Any) -> str:
        """Compatibility alias for chat(); used by services and tests.
 
        Args:
            prompt: User prompt.
            **kwargs: Provider-specific options.
 
        Returns:
            Model response string.
        """
        return self.chat(prompt, **kwargs)
 
 
class EchoProvider(AIProvider):
    """Local echo provider used for testing and defaults."""
 
    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
        """Initialize EchoProvider with a no-cost metadata preset."""
        meta = ProviderMetadata(
            name="Local Echo",
            default_model="echo",
            supported_models=["echo"],
            supports_streaming=False,
            supports_tools=False,
            supports_vision=False,
            max_context=1_000_000,
            cost_per_1k_tokens=0.0,
            avg_latency_ms=1,
            supports_rag=False,
        )
        super().__init__(
            provider_name="echo",
            model=model,
            provider_meta=meta,
            **kwargs,
        )
 
    def _send_impl(self, prompt: str) -> str:
        """Return the prompt prefixed with '(echo)'.
 
        Args:
            prompt: Validated prompt string.
 
        Returns:
            Echo string.
        """
        return f"(echo) {prompt}"
 
    def chat(self, prompt: str, **_kwargs: Any) -> str:
        """Return the prompt unchanged (no-op echo for chat).
 
        Args:
            prompt: User prompt.
            **_kwargs: Ignored.
 
        Returns:
            The original prompt string.
        """
        return prompt
 
    def ask(self, prompt: str, **_kwargs: Any) -> str:
        """Delegate to chat().
 
        Args:
            prompt: User prompt.
            **_kwargs: Ignored.
 
        Returns:
            The original prompt string.
        """
        return self.chat(prompt)
