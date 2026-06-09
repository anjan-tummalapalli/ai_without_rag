from __future__ import annotations

import time, uuid, logging, json
import dataclasses as _dc
from dataclasses import dataclass
from typing import Optional, Any
from abc import ABC, abstractmethod

from ai_cli.core.exceptions import ProviderRequestError, PromptValidationError
from ai_cli.core.resilience import RetryEngine
from ai_cli.utils.validation import ResponseValidator, HallucinationDetector
from ai_cli.telemetry.monitoring import ModelQualityMetrics
from ai_cli.providers.registry import register_chat_provider

logger = logging.getLogger("ai_gateway")

DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_MAX_PROMPT_LENGTH = 10_000


@dataclass(frozen=True)
class ProviderMetadata:
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


@dataclass
class ProviderConfig:
    """
    Configuration shared by provider implementations.
    """
    model: str | None = None
    api_key: str | None = None
    embedding_model: str | None = None
    timeout: float | None = None

class AIProvider(ABC):
    """
    Base AI provider with:
    - prompt validation
    - retrying transport (_send_impl)
    - robust response coercion
    - basic telemetry hooks
    """

    def __init__(
        self,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        provider_meta: Optional[ProviderMetadata] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        embedding_client: Optional[Any] = None,
        vector_db: Optional[Any] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        **kwargs,
    ) -> None:
        # Determine provider name safely: explicit arg > attribute on subclass > unknown
        resolved_name = provider_name or getattr(self, "provider_name", None) or "unknown"
        # store provider_name (allow subclasses to have set this already)
        try:
            self.provider_name = resolved_name
        except Exception:
            # if subclass made it read-only, we'll still use resolved_name in other places
            pass

        # Ensure there is provider metadata; do not hard-fail if not provided.
        if provider_meta is None:
            provider_meta = ProviderMetadata(
                name=resolved_name,
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
        self.provider_meta = provider_meta

        # Basic config validation
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            logger.debug("Invalid timeout %s provided, falling back to default %s", timeout, DEFAULT_TIMEOUT_SECONDS)
            timeout = DEFAULT_TIMEOUT_SECONDS

        self.timeout = int(timeout)
        # prefer explicit model, fallback to metadata default
        self.model = model or self.provider_meta.default_model
        self.api_key = kwargs.get("api_key")

        self.trace_id = str(uuid.uuid4())

        # Injectables (allow tests or external code to inject different retry engines/validators)
        self.retry_engine = kwargs.get("retry_engine") or RetryEngine()
        self.response_validator = kwargs.get("response_validator") or ResponseValidator()
        self.hallucination_detector = kwargs.get("hallucination_detector") or HallucinationDetector()
        self.metrics = kwargs.get("metrics") or ModelQualityMetrics(
            provider=resolved_name,
            model=self.model,
        )

        # RAG / storage
        self.embedding_client = embedding_client
        self.vector_db = vector_db
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -------------------------
    # Prompt validation
    # -------------------------
    def validate_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise PromptValidationError("prompt must be string")

        if "\x00" in prompt:
            raise PromptValidationError("prompt contains NUL byte")

        # Normalize whitespace and let prompt_corrector do heavier corrections
        prompt = prompt.strip()
        from ai_cli.core.prompt_corrector import prompt_corrector

        corrected = prompt_corrector(prompt)

        # ensure corrected is a non-empty trimmed string
        if not corrected or not isinstance(corrected, str) or not corrected.strip():
            raise PromptValidationError("prompt is empty")

        corrected = corrected.strip()

        if len(corrected) > DEFAULT_MAX_PROMPT_LENGTH:
            raise PromptValidationError(f"prompt too long (>{DEFAULT_MAX_PROMPT_LENGTH} characters)")

        return corrected

    # -------------------------
    # Abstract transport
    # -------------------------
    @abstractmethod
    def _send_impl(self, prompt: str) -> Any:
        """Implement transport-specific send. May return str, bytes, dict, list, numbers, dataclass, etc."""
        raise NotImplementedError()

    # -------------------------
    # Response coercion
    # -------------------------
    def _coerce_response_to_str(self, response: Any) -> str:
        if response is None:
            raise ProviderRequestError("empty response")

        # bytes-like handling
        if isinstance(response, (bytes, bytearray, memoryview)):
            try:
                return bytes(response).decode("utf-8", errors="replace")
            except Exception:
                return str(response)

        if isinstance(response, str):
            return response

        # dataclass -> dict
        if _dc.is_dataclass(response):
            try:
                return json.dumps(_dc.asdict(response), ensure_ascii=False)
            except Exception:
                # fall-through to generic
                pass

        if isinstance(response, (dict, list, tuple, int, float, bool)):
            try:
                return json.dumps(response, ensure_ascii=False)
            except Exception:
                # fallback to str representation
                return str(response)

        try:
            return str(response)
        except Exception as exc:
            raise ProviderRequestError(f"Unsupported response type: {type(response)}") from exc

    # -------------------------
    # Internal metric helpers
    # -------------------------
    def _record_latency(self, latency: float) -> None:
        try:
            # ModelQualityMetrics is expected to have total_latency_seconds; be tolerant if not.
            self.metrics.total_latency_seconds += latency
        except Exception:
            try:
                setattr(self.metrics, "total_latency_seconds", getattr(self.metrics, "total_latency_seconds", 0.0) + latency)
            except Exception:
                logger.debug("Could not record latency on metrics object")

        try:
            # Optional field
            if hasattr(self.metrics, "successes"):
                self.metrics.successes += 1
        except Exception:
            pass

    # -------------------------
    # Main send
    # -------------------------
    def send(self, prompt: str) -> str:
        validated = self.validate_prompt(prompt)

        # increment request count if available, fail-safe otherwise
        try:
            self.metrics.requests += 1
        except Exception:
            try:
                setattr(self.metrics, "requests", getattr(self.metrics, "requests", 0) + 1)
            except Exception:
                pass

        start = time.monotonic()
        try:
            raw = self.retry_engine.execute(lambda: self._send_impl(validated))

            result = self._coerce_response_to_str(raw)

            latency = time.monotonic() - start
            self._record_latency(latency)

            # validate response content (may raise a validation exception)
            try:
                self.response_validator.validate(result)
            except Exception as val_exc:
                # validation failures should be surfaced, but annotate metrics
                try:
                    self.metrics.failures += 1
                except Exception:
                    pass
                logger.debug("Response validation failed: %s", val_exc)
                raise

            # optional hallucination detection - don't hard-fail but log and record metric if available
            try:
                hallu = self.hallucination_detector.detect(result)
                if hallu:
                    logger.debug("Potential hallucination detected for provider=%s model=%s trace=%s", self.provider_name, self.model, self.trace_id)
                    if hasattr(self.metrics, "hallucinations"):
                        try:
                            self.metrics.hallucinations += 1
                        except Exception:
                            pass
            except Exception:
                # detection should not break request lifecycle
                logger.debug("Hallucination detection failed unexpectedly", exc_info=True)

            return result.strip()

        except Exception as exc:
            # increment failures metric (best-effort)
            try:
                self.metrics.failures += 1
            except Exception:
                pass
            logger.debug("Provider request failed provider=%s model=%s trace=%s error=%s", getattr(self, "provider_name", "unknown"), getattr(self, "model", "unknown"), self.trace_id, exc)
            raise ProviderRequestError(str(exc)) from exc
    
    def ask(self, prompt: str, **kwargs) -> str:
        """
        Backward-compatible chat interface.
        """
        return self.send(prompt)

    def chat(self, prompt: str) -> str:
        """
        Compatibility wrapper for legacy provider contract.
        """
        return self.send(prompt)

@register_chat_provider("echo")
class EchoProvider(AIProvider):
    """Local echo provider used for testing and defaults."""

    def __init__(self, model: Optional[str] = None, **kwargs: Any) -> None:
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
        super().__init__(provider_name="echo", model=model, provider_meta=meta, **kwargs)

    def _send_impl(self, prompt: str) -> str:
        # deterministic and cheap echo used for tests and defaults
        return f"(echo) {prompt}"
