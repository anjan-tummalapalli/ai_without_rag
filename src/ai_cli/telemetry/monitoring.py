from __future__ import annotations
import contextlib
import os
from dataclasses import dataclass

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


class Tracer:
    """Small wrapper for OpenTelemetry tracing. No-op if not installed."""

    def __init__(self, service_name: str = "ai_gateway"):
        """Initialize Tracer."""
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
        """Initialize Metrics."""
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
            "ai_gateway_latency_seconds",
            "Request latency seconds",
            ["provider"],
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


GLOBAL_TRACER = Tracer()
GLOBAL_METRICS = Metrics(port=int(os.getenv("PROMETHEUS_PORT", "8000")))