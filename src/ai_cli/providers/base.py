from __future__ import annotations

import time, uuid, logging, json
from dataclasses import dataclass
from typing import Optional, Any

from ai_cli.core.exceptions import (
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


class AIProvider:
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

        # Set provider_name attribute safely
        name_val = provider_name
        if name_val is None:
            try:
                name_val = self.provider_name
            except Exception:
                name_val = "unknown"

        try:
            self.provider_name = name_val
        except AttributeError:
            pass

        # FIX: do NOT hard-fail unless metadata is truly required
        self.provider_meta = provider_meta or ProviderMetadata(
            name=name_val or "unknown",
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

        self.timeout = timeout
        self.model = model or self.provider_meta.default_model
        self.api_key = kwargs.get("api_key")

        self.trace_id = str(uuid.uuid4())

        self.retry_engine = RetryEngine()
        self.response_validator = ResponseValidator()
        self.hallucination_detector = HallucinationDetector()
        self.metrics = ModelQualityMetrics(
            provider=name_val or "unknown",
            model=self.model,
        )

        # RAG
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

        from ai_cli.core.prompt_corrector import prompt_corrector

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
        raise NotImplementedError()

    def ask(self, prompt: str, **kwargs) -> str:
        """
        Public provider contract.
        Args:
            prompt: User input prompt.
            **kwargs: Provider-specific options.

        Returns:
            Model response as string.
        """
        return self.send(prompt)

    # -------------------------
    # Response coercion
    # -------------------------
    def _coerce_response_to_str(self, response: Any) -> str:
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


    def chat(self, prompt: str, **kwargs: Any) -> str:
        """
        Standard chat interface.

        Args:
            prompt: User prompt.
            **kwargs: Provider-specific options.

        Returns:
            Model response.
        """
        return self.send(prompt)


    def ask(self, prompt: str, **kwargs: Any) -> str:
        """
        Compatibility interface used by services/tests.

        Args:
            prompt: User prompt.
            **kwargs: Provider-specific options.

        Returns:
            Model response.
        """
        return self.chat(prompt, **kwargs)

class EchoProvider(AIProvider):
    """Local echo provider used for testing and defaults."""

    def __init__(self, model: str | None = None, **kwargs: Any) -> None:
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
            supports_rag=False,
        )
        super().__init__(provider_name="echo", model=model, provider_meta=meta, **kwargs)

    def _send_impl(self, prompt: str) -> str:
        """Return echoed prompt."""
        return f"(echo) {prompt}"


    def chat(self, prompt: str, **kwargs):
        return prompt

    def ask(self, prompt: str, **kwargs):
        return self.chat(prompt, **kwargs)
