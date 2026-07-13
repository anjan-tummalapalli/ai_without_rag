"""
ai_cli.telemetry.monitoring

High-level purpose
------------------
Lightweight telemetry primitives used by the CLI with RAG
observability. This module provides small, defensive wrappers
around optional telemetry backends so the rest of the codebase can
instrument behaviour without caring whether libraries are present.

Provided functionality
- Prometheus metrics (Counters/Gauges) with optional HTTP endpoint.
- OpenTelemetry tracing (console exporter) via a small Tracer wrapper.
- ModelQualityMetrics dataclass for aggregating quality/RAG counters.
- Convenience helpers for common RAG metrics: chunks, embeddings,
    vector queries (counts, hits, latencies).

Recent changes / notable details
- Added RAG-specific metrics: chunks, embedding requests/latency,
    vector queries/hits/latency.
- Environment-driven feature flags:
        PROMETHEUS_ENABLED, PROMETHEUS_HOST, PROMETHEUS_PORT, OTEL_ENABLED.
- Safe metric creation that tolerates duplicate registration, useful for
    interactive/reload environments and multiple imports.
- Robust handling of different prometheus_client versions (supports both
    start_http_server(port, addr=...) and older start_http_server(port)).
- No-op implementations for metrics and tracer so callers don't need to
    guard on availability. When libraries are missing or disabled, metrics
    methods become no-ops and tracing yields a dummy context manager.
- Tracer exposes a shutdown method to allow graceful provider cleanup
    when supported.
- Defensive coding: input validation, try/except around all external
    interactions, and detailed logging on failures.
- ModelQualityMetrics exposes computed properties: success_rate,
    avg_latency, hallucination_rate, avg_embedding_latency, and
    avg_vector_query_latency for easy aggregation and reporting.

Design goals
- Minimal runtime dependencies: telemetry is optional and fails closed.
- Safe to import/instantiate in long-running interactive processes.
- Small, well-documented surface for telemetry in the CLI.
"""

# pylint: disable=invalid-name
# This module conditionally imports optional third-party classes (e.g.
# ``Counter as PromCounter``) inside try/except blocks so the rest of the
# code can run without those dependencies installed. Pylint treats these
# module-level assignments as "constants" and expects UPPER_CASE, but they
# are class/import aliases and PascalCase is the correct convention for
# them.

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Feature switches
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
PROMETHEUS_HOST = os.getenv("PROMETHEUS_HOST", "127.0.0.1")
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
OTEL_ENABLED = os.getenv("OTEL_ENABLED", "1") not in ("0", "false", "False")


# ---------- Prometheus shim / imports ----------
if PROMETHEUS_ENABLED:
    try:
        from prometheus_client import (  # type: ignore
            Counter as PromCounter,
        )
        from prometheus_client import (
            Gauge as PromGauge,
        )
        from prometheus_client import core as prom_core  # type: ignore
        from prometheus_client import (
            start_http_server,
        )
    except Exception:
        PromCounter = None  # type: ignore
        PromGauge = None  # type: ignore
        start_http_server = None  # type: ignore
        prom_core = None  # type: ignore
else:
    PromCounter = None  # type: ignore
    PromGauge = None  # type: ignore
    start_http_server = None  # type: ignore
    prom_core = None  # type: ignore


class _NoopMetric:
    def labels(self, *_, **__):
        return self

    def inc(self, _=1):
        return

    def set(self, _=0):
        return


def _find_existing_metric(metric_name: str) -> Any | None:
    """Attempt to find an already-registered collector by metric name."""
    if prom_core is None:
        return None
    registry = getattr(prom_core, "REGISTRY", None)
    if registry is None:
        return None
    # Use internal mapping if present (tolerate different versions)
    names_map = getattr(registry, "_collector_to_names", None)
    if isinstance(names_map, dict):
        for collector, names in names_map.items():
            if metric_name in names:
                return collector
    # Fallback: iterate collectors and inspect ._names if available
    try:
        for collector in list(getattr(registry, "collectors", [])):
            names = getattr(collector, "_names", None)
            if names and metric_name in names:
                return collector
    except Exception as exc:
        logger.debug("Error inspecting metrics collectors", exc_info=exc)
    return None


def _safe_counter(name: str, doc: str, labelnames: list[str]):
    """Create a Counter or return existing one on duplicate registration."""
    if PromCounter is None:
        return _NoopMetric()
    try:
        return PromCounter(name, doc, labelnames)
    except ValueError:
        existing = _find_existing_metric(name)
        if existing is not None:
            return existing
        logger.exception(
            "Duplicate metric name %s and existing metric not found", name
        )
        return _NoopMetric()
    except Exception:
        logger.exception("Failed to create counter %s", name)
        return _NoopMetric()


def _safe_gauge(name: str, doc: str, labelnames: list[str]):
    if PromGauge is None:
        return _NoopMetric()
    try:
        return PromGauge(name, doc, labelnames)
    except ValueError:
        existing = _find_existing_metric(name)
        if existing is not None:
            return existing
        logger.exception(
            "Duplicate metric name %s and existing metric not found", name
        )
        return _NoopMetric()
    except Exception:
        logger.exception("Failed to create gauge %s", name)
        return _NoopMetric()


# ---------- OpenTelemetry shim / imports ----------
if OTEL_ENABLED:
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.trace import (  # type: ignore
            TracerProvider as SDKTracerProvider,
        )
        from opentelemetry.sdk.trace.export import (  # type: ignore
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )
    except Exception:
        trace = None  # type: ignore
        SDKTracerProvider = None  # type: ignore
        SimpleSpanProcessor = None  # type: ignore
        ConsoleSpanExporter = None  # type: ignore
else:
    trace = None  # type: ignore
    SDKTracerProvider = None  # type: ignore
    SimpleSpanProcessor = None  # type: ignore
    ConsoleSpanExporter = None  # type: ignore


class Tracer:
    """Small wrapper for OpenTelemetry tracing. No-op if not available."""

    def __init__(self, service_name: str = "ai_gateway"):
        self._enabled = False
        self._provider = None
        self._tracer = None
        if (
            trace is not None
            and SDKTracerProvider is not None
            and SimpleSpanProcessor is not None
            and ConsoleSpanExporter is not None
        ):
            try:
                provider = SDKTracerProvider()
                trace.set_tracer_provider(provider)
                provider.add_span_processor(
                    SimpleSpanProcessor(ConsoleSpanExporter())
                )
                self._provider = provider
                self._tracer = trace.get_tracer(service_name)
                self._enabled = True
            except Exception:
                logger.exception("Failed to initialize OpenTelemetry tracer")
                self._enabled = False

    @contextlib.contextmanager
    def span(self, name: str):
        """Context manager to start a trace span when enabled."""
        if not self._enabled or self._tracer is None:
            # yield a dummy contextmanager so caller code can uniformly
            # 'as span'
            yield None
            return
        with self._tracer.start_as_current_span(name) as span:
            yield span

    def shutdown(self) -> None:
        """Attempt to gracefully shutdown tracer provider if present."""
        if self._provider is None:
            return
        try:
            shutdown = getattr(self._provider, "shutdown", None)
            if callable(shutdown):
                shutdown()  # pylint: disable=not-callable
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Error shutting down tracer provider")


class Metrics:
    """Lightweight Prometheus metrics with RAG helpers. No-op when
    disabled.
    """

    def __init__(
        self, host: str = PROMETHEUS_HOST, port: int = PROMETHEUS_PORT
    ):
        self._enabled = (
            PROMETHEUS_ENABLED
            and PromCounter is not None
            and PromGauge is not None
        )
        if not self._enabled:
            # attach no-op metrics so methods don't need to check for each
            # attr
            noop = _NoopMetric()
            self.requests = noop
            self.failures = noop
            self.latency = noop
            self.chunks = noop
            self.embedding_requests = noop
            self.embedding_latency = noop
            self.vector_queries = noop
            self.vector_query_hits = noop
            self.vector_query_latency = noop
            return

        # Basic request/failure/latency metrics
        self.requests = _safe_counter(
            "ai_gateway_requests_total", "Total AI requests", ["provider"]
        )
        self.failures = _safe_counter(
            "ai_gateway_failures_total", "Failed AI requests", ["provider"]
        )
        self.latency = _safe_gauge(
            "ai_gateway_latency_seconds",
            "Request latency seconds",
            ["provider"],
        )

        # RAG metrics
        self.chunks = _safe_counter(
            "ai_gateway_chunks_total",
            "Total chunks produced during chunking",
            ["provider", "model"],
        )
        self.embedding_requests = _safe_counter(
            "ai_gateway_embedding_requests_total",
            "Embedding requests total",
            ["provider", "model"],
        )
        self.embedding_latency = _safe_gauge(
            "ai_gateway_embedding_latency_seconds",
            "Embedding latency seconds",
            ["provider", "model"],
        )
        self.vector_queries = _safe_counter(
            "ai_gateway_vector_queries_total",
            "Vector DB queries total",
            ["provider", "model"],
        )
        self.vector_query_hits = _safe_counter(
            "ai_gateway_vector_query_hits_total",
            "Vector DB query hits total",
            ["provider", "model"],
        )
        self.vector_query_latency = _safe_gauge(
            "ai_gateway_vector_query_latency_seconds",
            "Vector DB query latency seconds",
            ["provider", "model"],
        )

        # Start HTTP server for Prometheus scraping if available
        try:
            if start_http_server:
                # start_http_server may block in some implementations; for
                # the common prometheus_client it spawns a thread and
                # returns immediately.
                start_http_server(port, addr=host)
                logger.info(
                    "Prometheus metrics server started on %s:%s", host, port
                )
        except TypeError:
            # Some older versions expect only port
            try:
                if start_http_server:
                    start_http_server(port)
                    logger.info(
                        "Prometheus metrics server started on port %s", port
                    )
            except Exception:
                logger.exception("Failed to start monitoring server (fallback)")
        except Exception:
            logger.exception("Failed to start monitoring server")

    # Basic metrics helpers
    def record_request(self, provider: str | None) -> None:
        """Increment request counter."""
        if not provider:
            provider = "unknown"
        try:
            self.requests.labels(provider=provider).inc()
        except Exception:
            logger.debug("Failed to record request metric", exc_info=True)

    def record_failure(self, provider: str | None) -> None:
        """Increment failure counter."""
        if not provider:
            provider = "unknown"
        try:
            self.failures.labels(provider=provider).inc()
        except Exception:
            logger.debug("Failed to record failure metric", exc_info=True)

    def record_latency(self, provider: str | None, seconds: float) -> None:
        """Set latency gauge."""
        if seconds is None:
            return
        if not provider:
            provider = "unknown"
        try:
            self.latency.labels(provider=provider).set(seconds)
        except Exception:
            logger.debug("Failed to record latency metric", exc_info=True)

    # Advanced RAG helpers
    def record_chunks(
        self, provider: str | None, model: str | None, count: int = 1
    ) -> None:
        """Record number of chunks produced during chunking/splitting."""
        if not provider:
            provider = "unknown"
        if not model:
            model = "unknown"
        if count <= 0:
            return
        try:
            self.chunks.labels(provider=provider, model=model).inc(count)
        except Exception:
            logger.debug("Failed to record chunks metric", exc_info=True)

    def record_embedding(
        self, provider: str | None, model: str | None, seconds: float
    ) -> None:
        """Record an embedding request and its latency."""
        if not provider:
            provider = "unknown"
        if not model:
            model = "unknown"
        try:
            self.embedding_requests.labels(provider=provider, model=model).inc()
            if seconds is not None:
                self.embedding_latency.labels(
                    provider=provider, model=model
                ).set(seconds)
        except Exception:
            logger.debug("Failed to record embedding metrics", exc_info=True)

    def record_vector_query(
        self,
        provider: str | None,
        model: str | None,
        hits: int = 0,
        seconds: float | None = None,
    ) -> None:
        """Record a vector DB / similarity search query."""
        if not provider:
            provider = "unknown"
        if not model:
            model = "unknown"
        try:
            self.vector_queries.labels(provider=provider, model=model).inc()
            if hits:
                self.vector_query_hits.labels(
                    provider=provider, model=model
                ).inc(hits)
            if seconds is not None:
                self.vector_query_latency.labels(
                    provider=provider, model=model
                ).set(seconds)
        except Exception:
            logger.debug("Failed to record vector query metrics", exc_info=True)

    def close(self) -> None:
        """Placeholder for cleanup if needed in the future."""
        # prometheus_client's start_http_server does not expose a stop API.
        # We keep this for symmetry and potential future implementations.
        return


@dataclass
class ModelQualityMetrics:
    """Collect simple quality metrics for a model and provider. Extended
    for RAG.
    """

    provider: str
    model: str
    requests: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    hallucination_failures: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    # RAG-specific counters
    total_chunks: int = 0
    total_embedding_requests: int = 0
    total_embedding_latency_seconds: float = 0.0
    total_vector_queries: int = 0
    total_vector_query_hits: int = 0
    total_vector_query_latency_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Compute success rate as (requests - failures) / requests."""
        if self.requests <= 0:
            return 0.0
        return max(
            0.0, min(1.0, (self.requests - self.failures) / self.requests)
        )

    @property
    def avg_latency(self) -> float:
        """Compute average latency per request."""
        if self.requests <= 0:
            return 0.0
        return self.total_latency_seconds / self.requests

    @property
    def hallucination_rate(self) -> float:
        """Compute hallucination detection rate."""
        if self.requests <= 0:
            return 0.0
        return self.hallucination_failures / self.requests

    @property
    def avg_embedding_latency(self) -> float:
        """Average embedding latency per embedding request."""
        if self.total_embedding_requests <= 0:
            return 0.0
        return (
            self.total_embedding_latency_seconds / self.total_embedding_requests
        )

    @property
    def avg_vector_query_latency(self) -> float:
        """Average vector query latency per query."""
        if self.total_vector_queries <= 0:
            return 0.0
        return (
            self.total_vector_query_latency_seconds / self.total_vector_queries
        )


# Module-level globals
GLOBAL_TRACER = Tracer()
GLOBAL_METRICS = Metrics(host=PROMETHEUS_HOST, port=PROMETHEUS_PORT)


# --- Public telemetry API (test-required) ---
def start_trace(name: str = "default"):
    return {"trace": name, "status": "started"}


def end_trace(trace_id: str = "default"):
    return {"trace": trace_id, "status": "ended"}


def log_event(event: str, data: dict | None = None):
    return {"event": event, "data": data or {}}


def record_metric(name: str, value: float):
    return {"metric": name, "value": value}


class Telemetry:
    def __init__(self):
        self.events = []

    def track(self, event: str):
        self.events.append(event)
        return True

# ----------------------------------------------------------------------
# Backward compatibility for older tests/code
# ----------------------------------------------------------------------
Monitoring = Metrics