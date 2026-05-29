"""
ai_cli.telemetry.monitoring

High-level purpose
------------------
This module provides lightweight telemetry primitives used by the CLI:
- OpenTelemetry tracing via a small Tracer wrapper (console exporter).
- Prometheus metrics via a small Metrics wrapper (Counter/Gauge + optional HTTP endpoint).
- A simple ModelQualityMetrics dataclass for aggregating per-model/per-provider quality stats.

Design and behavior summary
---------------------------
- Optional dependencies:
    - If the 'opentelemetry' packages are not available, tracing becomes a no-op. The Tracer class
        exposes the same API (notably the span context manager) but will yield None and perform no tracing.
    - If 'prometheus_client' is not available, metrics become no-op. The Metrics class methods will
        silently return without raising, allowing the rest of the application to run without Prometheus.
    - Import and setup errors are caught and ignored so telemetry is optional and non-fatal.

- Tracer:
    - Intended for short-lived spans instrumenting operations.
    - When OpenTelemetry is present, a TracerProvider is installed and a ConsoleSpanExporter is used,
        so spans are printed to stdout/stderr (useful for local debugging).
    - Use Tracer.span(name) as a context manager to create a span; when disabled it yields None.

- Metrics:
    - Exposes three metric families (when prometheus_client is available):
        - Counter "ai_gateway_requests_total" with label "provider"
        - Counter "ai_gateway_failures_total" with label "provider"
        - Gauge "ai_gateway_latency_seconds" with label "provider"
    - Convenience methods:
        - record_request(provider: str)
        - record_failure(provider: str)
        - record_latency(provider: str, seconds: float)
    - If start_http_server is available, the Metrics constructor will attempt to start an HTTP endpoint
        on the port given (default 8000). The port can be overridden via the PROMETHEUS_PORT environment
        variable.

- ModelQualityMetrics:
    - Lightweight dataclass for aggregating per-provider/per-model statistics:
        - requests, failures, total_latency_seconds, hallucination_failures,
            total_prompt_tokens, total_completion_tokens.
    - Provides computed read-only properties:
        - success_rate: fraction of requests that did not fail (0.0 if no requests)
        - avg_latency: average latency in seconds (0.0 if no requests)
        - hallucination_rate: fraction of requests flagged as hallucinations (0.0 if no requests)

Exposed module-level targets
----------------------------
- GLOBAL_TRACER: an instance of Tracer constructed at import time (no-op if OTEL missing).
- GLOBAL_METRICS: an instance of Metrics constructed at import time using PROMETHEUS_PORT or 8000.

Operational notes
-----------------
- The module intentionally swallows errors during telemetry setup to avoid impacting primary
    application behavior; it is best-effort telemetry only.
- The console exporter and Prometheus HTTP server are convenient for local development and
    simple deployments. For production-grade tracing/metrics, replace or augment the exporter/collector
    configuration as appropriate.
"""
from __future__ import annotations
import contextlib
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, start_http_server # type: ignore
except Exception:  # optional
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore

try:
    from opentelemetry import trace # type: ignore
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider # type: ignore
    from opentelemetry.sdk.trace.export import ( # type: ignore
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
        except Exception as exc:
            logger.warning(
            "Failed to start monitoring server: %s",
            exc,
            exc_info=True,
    )

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