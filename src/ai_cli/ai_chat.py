"""
Lightweight AI gateway helpers.

This module provides small, opt-in operational components and a
simple provider framework. Optional dependencies are permissive:
missing packages fall back to no-op or pure-Python implementations
so the module remains usable without extras.

To enable richer behavior install the extras:
    pip install redis prometheus_client opentelemetry-api \
        opentelemetry-sdk aiohttp
"""

from __future__ import annotations

import argparse, asyncio, contextlib, functools, inspect, logging, os, sys
import re, time, uuid, random, threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional, Final, TypeVar

try:
    import redis
except Exception:  # optional
    redis = None  # type: ignore

try:
    from prometheus_client import Counter, Gauge, start_http_server
except Exception:  # optional
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from opentelemetry.sdk.trace.export import (
        SimpleSpanProcessor,
        ConsoleSpanExporter,
    )
except Exception:  # optional
    trace = None  # type: ignore

# -----------------------------------------------------------------------------
# Operational helpers (cache, retry, circuit breaker, rate limiter,
# tracing)
# -----------------------------------------------------------------------------

class Cache:
    """Simple cache: Redis-backed when available, else in-memory LRU.

    Purpose:
        Provide namespaced cache get/set semantics with optional Redis
        backend and a local LRU fallback.

    Args:
        namespace (str): Prefix used for keys stored in Redis or local cache.
        max_entries (int): Maximum number of entries in local in-memory cache.
    """

    def __init__(self, namespace: str = "ai_gateway", max_entries: int = 1024):
        """Initialize Cache.

        Args:
            namespace (str): Namespace prefix for keys.
            max_entries (int): Maximum local cache entries.
        """
        self.namespace = namespace
        self.max_entries = max_entries
        self._local_cache: dict[str, Any] = {}
        self._local_order: Deque[str] = deque()
        if redis is not None:
            try:
                url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self._redis = redis.Redis.from_url(url)
                self._redis.ping()
            except Exception:
                self._redis = None
        else:
            self._redis = None

    def _key(self, k: str) -> str:
        """Return a namespaced key.

        Args:
            k (str): Original key.

        Returns:
            str: Namespaced key string.
        """
        return f"{self.namespace}:{k}"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from cache.

        Args:
            key (str): Key to retrieve.

        Returns:
            Optional[Any]: Stored value or None if missing.
        """
        if self._redis:
            try:
                val = self._redis.get(self._key(key))
                return None if val is None else val
            except Exception:
                return None
        return self._local_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value in cache with optional TTL.

        Args:
            key (str): Key to set.
            value (Any): Value to store.
            ttl (Optional[int]): Time-to-live in seconds for Redis backend.
        """
        if self._redis:
            try:
                self._redis.set(self._key(key), value, ex=ttl)
                return
            except Exception:
                pass
        if key in self._local_cache:
            try:
                self._local_order.remove(key)
            except ValueError:
                pass
        self._local_cache[key] = value
        self._local_order.appendleft(key)
        while len(self._local_order) > self.max_entries:
            old = self._local_order.pop()
            self._local_cache.pop(old, None)


class AsyncRetryEngine:
    """Async retry executor with exponential backoff.

    Purpose:
        Execute asynchronous or awaitable-producing callables with retries
        and exponential backoff.

    Args:
        max_attempts (int): Maximum retry attempts.
        base_delay (float): Base delay in seconds for backoff.
    """

    def __init__(self, max_attempts: int = 3, base_delay: float = 0.5):
        """Initialize AsyncRetryEngine.

        Args:
            max_attempts (int): Maximum retry attempts.
            base_delay (float): Base delay in seconds.
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def execute(self, coro_fn: Callable[[], Any]) -> Any:
        """Execute the provided coroutine-producing callable with retries.

        Args:
            coro_fn (Callable[[], Any]): Callable that returns an awaitable
                or value.

        Returns:
            Any: Result returned by awaited coroutine or the callable.

        Raises:
            Exception: Re-raises last exception if all attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                res = coro_fn()
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception as exc:
                last_exc = exc
                sleep_time = self.base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time + (0.1 * attempt))
        assert last_exc is not None
        raise last_exc


class CircuitBreaker:
    """Minimal thread-safe circuit breaker.

    Purpose:
        Track failures and open the circuit for a timeout when threshold is
        exceeded to prevent cascading failures.

    Args:
        threshold (int): Failure count threshold to open circuit.
        timeout (int): Seconds to keep circuit open.
    """

    def __init__(self, threshold: int = 5, timeout: int = 30):
        """Initialize CircuitBreaker.

        Args:
            threshold (int): Failure threshold before opening circuit.
            timeout (int): Open-circuit timeout in seconds.
        """
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.open_until = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """Check whether requests are allowed.

        Returns:
            bool: True if circuit is closed, False if open.
        """
        with self._lock:
            return time.time() >= self.open_until

    def success(self) -> None:
        """Record a successful call and reset failure tracking."""
        with self._lock:
            self.failures = 0
            self.open_until = 0.0

    def failure(self) -> None:
        """Record a failure and open circuit if threshold exceeded."""
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.open_until = time.time() + self.timeout

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a callable with circuit breaker checks."""

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            if not self.allow():
                raise RuntimeError("circuit open")
            try:
                result = fn(*args, **kwargs)
            except Exception:
                self.failure()
                raise
            else:
                self.success()
                return result

        return _wrapped


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, capacity: int = 10, rate_per_second: float = 1.0):
        """Initialize RateLimiter.

        Args:
            capacity (int): Token capacity.
            rate_per_second (float): Token refill rate per second.
        """
        self.capacity = capacity
        self.tokens = float(capacity)
        self.rate = rate_per_second
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + add)

    def allow(self, cost: float = 1.0) -> bool:
        """Attempt to consume tokens to allow an action.

        Args:
            cost (float): Token cost for the action.

        Returns:
            bool: True if tokens were available and consumed.
        """
        with self._lock:
            self._refill()
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a callable to enforce rate limiting."""

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            if not self.allow():
                raise RuntimeError("rate limited")
            return fn(*args, **kwargs)

        return _wrapped


class StreamConsumer:
    """
    Helper to adapt synchronous providers into streaming-like behavior.

    Purpose:
        Provide a streaming interface via either provider.send_stream or by
        running provider.send in a background thread and emitting tokens.
    """

    def __init__(self, provider: Any):
        """Initialize StreamConsumer.

        Args:
            provider (Any): Provider implementing send or send_stream.
        """
        self.provider = provider

    def stream(self, prompt: str, on_token: Callable[[str], None]) -> None:
        """Start streaming tokens for a prompt."""
        if hasattr(self.provider, "send_stream"):
            return self.provider.send_stream(prompt, on_token)

        def _worker():
            resp = self.provider.send(prompt)
            for token in resp.split():
                on_token(token + " ")
            on_token("")  # sentinel

        thr = threading.Thread(target=_worker, daemon=True)
        thr.start()


class Tracer:
    """Small wrapper for OpenTelemetry tracing. No-op if not installed."""

    def __init__(self, service_name: str = "ai_gateway"):
        """Initialize Tracer.

        Args:
            service_name (str): Service name for tracer.
        """
        self._enabled = False
        if trace is not None:
            try:
                provider = SDKTracerProvider()
                trace.set_tracer_provider(provider)
                tracer = trace.get_tracer(__name__)
                provider.add_span_processor(
                    SimpleSpanProcessor(ConsoleSpanExporter())
                )
                self._tracer = tracer
                self._enabled = True
            except Exception:
                self._enabled = False

    @contextlib.contextmanager
    def span(self, name: str):
        """Context manager to start a trace span when enabled."""
        if not self._enabled:
            yield None
            return
        with self._tracer.start_as_current_span(name) as span:
            yield span


class Metrics:
    """Lightweight Prometheus metrics. No-op if prometheus_client missing."""

    def __init__(self, port: int = 8000):
        """Initialize Metrics.

        Args:
            port (int): HTTP port to expose Prometheus metrics.
        """
        self._enabled = Counter is not None
        if not self._enabled:
            return
        self.requests = Counter(
            "ai_gateway_requests_total", "Total AI requests", ["provider"]
        )
        self.failures = Counter(
            "ai_gateway_failures_total", "Failed AI requests", ["provider"]
        )
        self.latency = Gauge(
            "ai_gateway_latency_seconds", "Request latency seconds", ["provider"]
        )
        try:
            if start_http_server:
                start_http_server(port)
        except Exception:
            pass

    def record_request(self, provider: str) -> None:
        """Increment request counter."""
        if not self._enabled:
            return
        self.requests.labels(provider=provider).inc()

    def record_failure(self, provider: str) -> None:
        """Increment failure counter."""
        if not self._enabled:
            return
        self.failures.labels(provider=provider).inc()

    def record_latency(self, provider: str, seconds: float) -> None:
        """Set latency gauge."""
        if not self._enabled:
            return
        self.latency.labels(provider=provider).set(seconds)


class SecretManager:
    """Resolve secret from environment or a file."""

    @staticmethod
    def get_secret(name: str, file_path: Optional[str] = None) -> Optional[str]:
        """Obtain a secret value."""
        val = os.getenv(name)
        if val:
            return val
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    return fh.read().strip()
            except Exception:
                return None
        return None


def is_kubernetes() -> bool:
    """Heuristic detection for running inside Kubernetes."""
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return True
    return os.path.exists(
        "/var/run/secrets/kubernetes.io/serviceaccount/token"
    )


GLOBAL_CACHE = Cache()
GLOBAL_TRACER = Tracer()
GLOBAL_METRICS = Metrics(
    port=int(os.getenv("PROMETHEUS_PORT", "8000"))
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ai_gateway")

# -----------------------------------------------------------------------------
# Constants, types, exceptions
# -----------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS: Final[int] = 60
DEFAULT_MAX_PROMPT_LENGTH: Final[int] = 10_000
MIN_RESPONSE_LENGTH: Final[int] = 5

T = TypeVar("T")


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


# -----------------------------------------------------------------------------
# Provider metadata and registry (trimmed to implemented providers)
# -----------------------------------------------------------------------------

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


PROVIDERS: dict[str, ProviderMetadata] = {
    "openai": ProviderMetadata(
        name="OpenAI",
        default_model="gpt-5.5",
        supported_models=["gpt-5.5", "gpt-4.1", "gpt-4o"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=1_000_000,
        cost_per_1k_tokens=0.01,
        avg_latency_ms=800,
    ),
    "perplexity": ProviderMetadata(
        name="Perplexity AI",
        default_model="sonar-pro",
        supported_models=["sonar", "sonar-pro"],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.009,
        avg_latency_ms=850,
    ),
    "deepseek": ProviderMetadata(
        name="DeepSeek",
        default_model="deepseek-chat",
        supported_models=[
            "deepseek-chat",
            "deepseek-coder",
            "deepseek-reasoner",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.003,
        avg_latency_ms=500,
    ),
    "groq": ProviderMetadata(
        name="Groq",
        default_model="llama-3.3-70b",
        supported_models=["llama-3.3-70b", "mixtral-8x7b"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.002,
        avg_latency_ms=300,
    ),
    "openrouter": ProviderMetadata(
        name="OpenRouter",
        default_model="openai/gpt-4o",
        supported_models=[
            "openai/gpt-4o",
            "anthropic/claude-3.5-sonnet",
            "google/gemini-2.5-pro",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=200_000,
        cost_per_1k_tokens=0.005,
        avg_latency_ms=700,
    ),
    "together": ProviderMetadata(
        name="Together AI",
        default_model="meta-llama/Llama-3-70b-chat-hf",
        supported_models=[
            "meta-llama/Llama-3-70b-chat-hf",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.004,
        avg_latency_ms=650,
    ),
    "fireworks": ProviderMetadata(
        name="Fireworks AI",
        default_model=(
            "accounts/fireworks/models/llama-v3p1-70b-instruct"
        ),
        supported_models=[
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
        ],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.004,
        avg_latency_ms=550,
    ),
    "xai": ProviderMetadata(
        name="xAI Grok",
        default_model="grok-3",
        supported_models=["grok-3", "grok-3-mini"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        cost_per_1k_tokens=0.011,
        avg_latency_ms=750,
    ),
    "google": ProviderMetadata(
        name="Google",
        default_model="gemini-pro",
        supported_models=[
            "gemini-pro",
            "gemini-1.5",
            "gemini-1.0",
            "gemini-mini",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=1_000_000,
        cost_per_1k_tokens=0.012,
        avg_latency_ms=600,
    ),
    "echo": ProviderMetadata(
        name="Local Echo",
        default_model="echo",
        supported_models=["echo"],
        supports_streaming=False,
        supports_tools=False,
        supports_vision=False,
        max_context=1_000_000,
        cost_per_1k_tokens=0.0,
        avg_latency_ms=1,
    ),
}

AVAILABLE_MODELS: dict[str, list[str]] = {
    provider: metadata.supported_models
    for provider, metadata in PROVIDERS.items()
}

# -----------------------------------------------------------------------------
# Metrics and detectors
# -----------------------------------------------------------------------------

@dataclass
class ModelQualityMetrics:
    """Collect simple quality metrics for a model and provider."""
    provider: str
    model: str
    requests: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    hallucination_failures: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def success_rate(self) -> float:
        """Compute success rate as (requests - failures) / requests."""
        if self.requests == 0:
            return 0.0
        return (self.requests - self.failures) / self.requests

    @property
    def avg_latency(self) -> float:
        """Compute average latency per request."""
        if self.requests == 0:
            return 0.0
        return self.total_latency_seconds / self.requests

    @property
    def hallucination_rate(self) -> float:
        """Compute hallucination detection rate."""
        if self.requests == 0:
            return 0.0
        return self.hallucination_failures / self.requests


@dataclass
class HallucinationResult:
    """Result of hallucination risk evaluation."""
    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


class HallucinationDetector:
    """Heuristic-based hallucination risk estimator (not a verifier)."""

    SUSPICIOUS_PATTERNS: Final[list[str]] = [
        r"100% accurate",
        r"guaranteed",
        r"always works",
        r"never fails",
        r"trust me",
    ]

    def evaluate(self, response: str) -> HallucinationResult:
        """Evaluate response for hallucination risk."""
        score = 0.0
        reasons: list[str] = []

        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            score += 0.4
            reasons.append("response too short")

        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                score += 0.2
                reasons.append(f"suspicious phrase: {pattern}")

        if "TODO" in response:
            score += 0.3
            reasons.append("placeholder content detected")

        score = min(score, 1.0)
        return HallucinationResult(score=score, passed=score < 0.5,
                                   reasons=reasons)


class ResponseValidator:
    """Simple response validation helper."""

    def validate(self, response: str) -> None:
        """Validate the response string.

        Raises:
            ResponseValidationError: If response is empty or too short.
        """
        if not response:
            raise ResponseValidationError("empty response")
        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            raise ResponseValidationError("response too short")


# -----------------------------------------------------------------------------
# Retry engine (sync)
# -----------------------------------------------------------------------------

class RetryEngine:
    """Sync retry executor with exponential backoff."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0) -> None:
        """Initialize RetryEngine."""
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def execute(self, func: Callable[[], T]) -> T:
        """Execute callable with retries."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                sleep_time = self.base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.5)
                logger.warning(
                    "retry_attempt=%s sleep=%s error=%s",
                    attempt,
                    sleep_time,
                    exc,
                )
                time.sleep(sleep_time + jitter)
        raise last_error  # type: ignore[misc]


# -----------------------------------------------------------------------------
# AI Provider base and concrete providers
# -----------------------------------------------------------------------------

class AIProvider:
    """Abstract base for provider integrations."""

    def __init__(
        self,
        provider_name: str,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize AIProvider base."""
        if provider_name not in PROVIDERS:
            raise ProviderConfigurationError(
                f"Unknown provider metadata '{provider_name}'"
            )
        self.provider_name = provider_name
        self.timeout = timeout
        provider_meta = PROVIDERS[provider_name]
        self.model = (model or provider_meta.default_model)
        self.trace_id = str(uuid.uuid4())
        self.retry_engine = RetryEngine()
        self.response_validator = ResponseValidator()
        self.hallucination_detector = HallucinationDetector()
        self.metrics = ModelQualityMetrics(
            provider=provider_name, model=self.model
        )

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

    def _send_impl(self, prompt: str) -> str:
        """Provider-specific implementation must override."""
        raise NotImplementedError

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
            response = self.retry_engine.execute(
                lambda: self._send_impl(validated_prompt)
            )
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
        super().__init__(provider_name="echo", model=model)

    def _send_impl(self, prompt: str) -> str:
        """Return echoed prompt."""
        return f"(echo) {prompt}"


class OpenAIProvider(AIProvider):
    """Concrete provider using OpenAI SDK."""

    def __init__(self, model: str | None = None) -> None:
        """Initialize OpenAIProvider."""
        super().__init__(provider_name="openai", model=model)

    def _send_impl(self, prompt: str) -> str:
        """Send prompt to OpenAI API and return content."""
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError("Install openai package") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"OpenAI request failed: {exc}"
            ) from exc

        usage = getattr(response, "usage", None)
        if usage:
            self.metrics.total_prompt_tokens += getattr(
                usage, "prompt_tokens", 0
            )
            self.metrics.total_completion_tokens += getattr(
                usage, "completion_tokens", 0
            )

        try:
            return response.choices[0].message.content
        except Exception as exc:
            raise ResponseValidationError(
                "Invalid response structure"
            ) from exc


class OpenAICompatibleProvider(AIProvider):
    """Generic provider for OpenAI-compatible APIs."""

    api_base_url: str = ""
    api_key_env: str = ""

    def __init__(self, provider_name: str, model: str | None = None) -> None:
        """Initialize OpenAICompatibleProvider."""
        super().__init__(provider_name=provider_name, model=model)

    def _send_impl(self, prompt: str) -> str:
        """Send prompt using OpenAI SDK against configured base_url."""
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install OpenAI SDK: pip install openai"
            ) from exc

        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderConfigurationError(f"{self.api_key_env} not set")

        client = OpenAI(api_key=api_key, base_url=self.api_base_url)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"{self.provider_name} request failed: {exc}"
            ) from exc

        usage = getattr(response, "usage", None)
        if usage:
            self.metrics.total_prompt_tokens += getattr(
                usage, "prompt_tokens", 0
            )
            self.metrics.total_completion_tokens += getattr(
                usage, "completion_tokens", 0
            )

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ResponseValidationError(
                "Invalid response structure"
            ) from exc
        if not content or not isinstance(content, str):
            raise ResponseValidationError("Empty response")
        return content.strip()


class PerplexityProvider(OpenAICompatibleProvider):
    """Provider for Perplexity API."""
    api_base_url = "https://api.perplexity.ai"
    api_key_env = "PERPLEXITY_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize PerplexityProvider."""
        super().__init__(provider_name="perplexity", model=model)


class DeepSeekProvider(OpenAICompatibleProvider):
    """Provider for DeepSeek API."""
    api_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize DeepSeekProvider."""
        super().__init__(provider_name="deepseek", model=model)


class GroqProvider(OpenAICompatibleProvider):
    """Provider for Groq API."""
    api_base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize GroqProvider."""
        super().__init__(provider_name="groq", model=model)


class OpenRouterProvider(OpenAICompatibleProvider):
    """Provider for OpenRouter API."""
    api_base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize OpenRouterProvider."""
        super().__init__(provider_name="openrouter", model=model)


class TogetherProvider(OpenAICompatibleProvider):
    """Provider for Together AI API."""
    api_base_url = "https://api.together.xyz/v1"
    api_key_env = "TOGETHER_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize TogetherProvider."""
        super().__init__(provider_name="together", model=model)


class FireworksProvider(OpenAICompatibleProvider):
    """Provider for Fireworks AI API."""
    api_base_url = "https://api.fireworks.ai/inference/v1"
    api_key_env = "FIREWORKS_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize FireworksProvider."""
        super().__init__(provider_name="fireworks", model=model)


class XAIProvider(OpenAICompatibleProvider):
    """Provider for xAI Grok API."""
    api_base_url = "https://api.x.ai/v1"
    api_key_env = "XAI_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize XAIProvider."""
        super().__init__(provider_name="xai", model=model)


class GeminiProvider(OpenAICompatibleProvider):
    """Google Gemini-compatible provider (uses OpenAI shim)."""
    api_base_url = "https://api.gemini.google/v1"
    api_key_env = "GEMINI_API_KEY"

    def __init__(self, model: str | None = None) -> None:
        """Initialize GeminiProvider."""
        super().__init__(provider_name="gemini", model=model)


# -----------------------------------------------------------------------------
# Provider factory
# -----------------------------------------------------------------------------

PROVIDER_MAP: dict[str, type[AIProvider]] = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "perplexity": PerplexityProvider,
    "deepseek": DeepSeekProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "together": TogetherProvider,
    "fireworks": FireworksProvider,
    "xai": XAIProvider,
    "gemini": GeminiProvider,
}


def build_provider(name: str, model: str | None = None) -> AIProvider:
    """Factory that constructs a provider instance by name."""
    normalized_name = name.lower()
    try:
        provider_class = PROVIDER_MAP[normalized_name]
    except KeyError as exc:
        raise ProviderConfigurationError(f"Unknown provider '{name}'") from exc
    return provider_class(model=model)


# -----------------------------------------------------------------------------
# High-level API
# -----------------------------------------------------------------------------

def ask(provider: str, prompt: str, model: str | None = None) -> str:
    """High-level convenience function to ask a provider a prompt."""
    if not isinstance(provider, str):
        return "[ERROR] provider must be string"
    provider = provider.strip().lower()
    if not provider:
        return "[ERROR] provider is empty"
    if provider not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        return (
            f"[ERROR] Invalid provider '{provider}'. Available providers: "
            f"{available}"
        )

    if not isinstance(prompt, str):
        return "[ERROR] prompt must be string"
    prompt = prompt.strip()
    if not prompt:
        return "[ERROR] Invalid prompt"

    if model is not None:
        if not isinstance(model, str):
            return "[ERROR] model must be string"
        model = model.strip()
        if not model:
            return "[ERROR] model is empty"
        supported_models = PROVIDERS[provider].supported_models
        if model not in supported_models:
            supported = ", ".join(supported_models)
            return (
                f"[ERROR] Invalid model '{model}' for provider "
                f"'{provider}'. Supported models: {supported}"
            )

    try:
        ai_provider = build_provider(name=provider, model=model)
        response = ai_provider.send(prompt)

        if not isinstance(response, str):
            logger.error(
                "invalid_response_type provider=%s type=%s",
                provider,
                type(response).__name__,
            )
            return "[ERROR] Invalid response type"
        response = response.strip()
        if not response:
            return "[ERROR] Empty response"
        return response
    except AIProviderError as exc:
        logger.error("ai_provider_error provider=%s error=%s", provider, exc)
        return f"[ERROR] {exc}"
    except Exception as exc:
        logger.exception(
            "unexpected_ask_failure provider=%s model=%s error=%s",
            provider,
            model,
            exc,
        )
        return "[ERROR] Unexpected internal error. Check logs."


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(prog="ai_chat", description=(
        "Enterprise AI Gateway CLI"
    ))
    parser.add_argument("-p", "--provider", default="echo",
                        help="Provider name")
    parser.add_argument("-m", "--prompt", help="Prompt text")
    parser.add_argument("-M", "--model", help="Model override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint main function."""
    args = parse_args(argv)
    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2
    response = ask(provider=args.provider, prompt=prompt, model=args.model)
    print(response)
    return 0


__all__ = [
    "ask",
    "PROVIDERS",
    "AIProvider",
    "RetryEngine",
    "build_provider",
    "OpenAIProvider",
    "ProviderMetadata",
    "ModelQualityMetrics",
    "HallucinationDetector",
]

if __name__ == "__main__":
    raise SystemExit(main())
