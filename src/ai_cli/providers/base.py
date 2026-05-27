"""Base classes for AI provider integrations.

This module defines an abstract AIProvider used by provider-specific
integrations. It centralizes prompt validation, retry handling,
response coercion, response validation, hallucination detection, and
model quality metrics so individual providers only implement transport
and API specifics. The concrete EchoProvider provides a simple local
echo implementation for testing.

End result: calling send(prompt) returns a validated, coerced string
response (or raises ProviderRequestError on failure) while updating
metrics and logging relevant events.
"""

from __future__ import annotations
import time
import uuid
import logging
import json
from dataclasses import dataclass

from ai_cli.core.exceptions import (
    ProviderConfigurationError,
    ProviderRequestError,
    PromptValidationError,
)
from ai_cli.core.resilience import RetryEngine
from ai_cli.utils.validation import ResponseValidator, HallucinationDetector
from ai_cli.telemetry.monitoring import ModelQualityMetrics

logger = logging.getLogger("ai_gateway")

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_PROMPT_LENGTH = 10_000


@dataclass(frozen=True)
class ProviderMetadata:
    """Metadata describing a provider and its capabilities."""

    name: str
    default_model: str
    supported_models: list[str]
    supports_streaming: bool
    supports_tools: bool
    supports_vision: bool
    max_context: int
    cost_per_1k_tokens: float
    avg_latency_ms: int


class AIProvider:
    """Abstract base for provider integrations."""

    def __init__(
        self,
        provider_name: str,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        provider_meta: ProviderMetadata | None = None,
    ) -> None:
        """Initialize AIProvider base."""
        if provider_meta is None:
            raise ProviderConfigurationError("Provider metadata is required")

        self.provider_name = provider_name
        self.timeout = timeout
        self.model = model or provider_meta.default_model
        self.trace_id = str(uuid.uuid4())
        self.retry_engine = RetryEngine()
        self.response_validator = ResponseValidator()
        self.hallucination_detector = HallucinationDetector()
        self.metrics = ModelQualityMetrics(
            provider=provider_name, model=self.model
        )
        self._provider_meta = provider_meta

    def validate_prompt(self, prompt: str) -> str:
        """Validate and sanitize a prompt string."""
        if not isinstance(prompt, str):
            raise PromptValidationError("prompt must be string")
        prompt = prompt.strip()
        if not prompt:
            raise PromptValidationError("prompt is empty")
        if len(prompt) > DEFAULT_MAX_PROMPT_LENGTH:
            raise PromptValidationError("prompt exceeds maximum length")
        if "\x00" in prompt:
            raise PromptValidationError("prompt contains NUL byte")
        sanitized = "".join(
            ch for ch in prompt if ch in ("\n", "\t") or ord(ch) >= 32
        )
        return sanitized

    def _send_impl(self, _prompt: str) -> str:
        """Provider-specific implementation must override."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _send_impl()"
        )

    def _coerce_response_to_str(self, response) -> str:
        """Handle non-string responses from providers and coerce to str."""
        if response is None:
            raise ProviderRequestError(
                f"{self.provider_name} returned empty response (None)"
            )
        if isinstance(response, str):
            return response
        if isinstance(response, bytes):
            try:
                return response.decode("utf-8")
            except Exception:
                return response.decode("latin-1", errors="ignore")
        if isinstance(response, (dict, list, tuple, int, float, bool)):
            try:
                return json.dumps(response, ensure_ascii=False)
            except Exception as exc:
                logger.warning(
                    "response_serialization_failed provider=%s error=%s",
                    self.provider_name,
                    exc,
                )
                # Fallback to str()
                return str(response)
        # Best-effort fallback for arbitrary objects
        try:
            return str(response)
        except Exception as exc:
            logger.exception(
                "response_coercion_failed provider=%s trace_id=%s error=%s",
                self.provider_name,
                self.trace_id,
                exc,
            )
            raise ProviderRequestError(
                f"{self.provider_name} returned an unsupported response type: {type(response)}"
            ) from exc

    def send(self, prompt: str) -> str:
        """High-level send that validates prompt, runs retries, and checks."""
        validated_prompt = self.validate_prompt(prompt)
        self.metrics.requests += 1
        start_time = time.monotonic()

        logger.info(
            "provider_request provider=%s model=%s trace_id=%s",
            self.provider_name,
            self.model,
            self.trace_id,
        )

        try:
            raw_response = self.retry_engine.execute(
                lambda: self._send_impl(validated_prompt)
            )

            # Coerce non-string responses (e.g., dicts, bytes, numbers) into a string
            response = self._coerce_response_to_str(raw_response)

            duration = time.monotonic() - start_time
            self.metrics.total_latency_seconds += duration
            self.response_validator.validate(response)

            hallucination = self.hallucination_detector.evaluate(response)
            if not hallucination.passed:
                self.metrics.hallucination_failures += 1
                logger.warning(
                    "hallucination_detected provider=%s score=%s reasons=%s",
                    self.provider_name,
                    hallucination.score,
                    hallucination.reasons,
                )
            return response.strip()
        except Exception as exc:
            self.metrics.failures += 1
            logger.exception(
                "provider_error provider=%s trace_id=%s error=%s",
                self.provider_name,
                self.trace_id,
                exc,
            )
            raise ProviderRequestError(
                f"{self.provider_name} request failed: {exc}"
            ) from exc


class EchoProvider(AIProvider):
    """Local echo provider used for testing and defaults."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize EchoProvider."""
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
        )
        super().__init__(provider_name="echo", model=model, provider_meta=meta)

    def _send_impl(self, prompt: str) -> str:
        """Return echoed prompt."""
        return f"(echo) {prompt}"
