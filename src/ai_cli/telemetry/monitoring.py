"""
ai_cli.telemetry.monitoring

High-level purpose
------------------
This module provides lightweight telemetry primitives used by the CLI and now
includes additional observability primitives for Advanced Retrieval-Augmented
Generation (RAG) workflows:

- OpenTelemetry tracing via a small Tracer wrapper (console exporter).
- Prometheus metrics via a small Metrics wrapper (Counter/Gauge + optional HTTP endpoint).
- A simple ModelQualityMetrics dataclass for aggregating per-model/per-provider quality stats.

Advanced RAG additions
----------------------
To help observe RAG-specific steps, Metrics now exposes (when prometheus_client
is available) additional metric families instrumenting core RAG pipeline stages:
- Chunking:
    - Counter "ai_gateway_chunks_total" with labels ("provider","model")
      Records number of text chunks produced during chunking/splitting.
- Embeddings:
    - Counter "ai_gateway_embedding_requests_total" with labels ("provider","model")
      Counts embedding requests.
    - Gauge "ai_gateway_embedding_latency_seconds" with labels ("provider","model")
      Records embedding latency (seconds).
- Vector DB / Similarity Search:
    - Counter "ai_gateway_vector_queries_total" with labels ("provider","model")
    - Counter "ai_gateway_vector_query_hits_total" with labels ("provider","model")
    - Gauge "ai_gateway_vector_query_latency_seconds" with labels ("provider","model")

Convenience methods are provided on Metrics:
- record_chunks(provider, model, count: int)
- record_embedding(provider, model, seconds: float)
- record_vector_query(provider, model, hits: int, seconds: float)

Design notes
------------
- Optional dependencies:
    - If the 'opentelemetry' packages are not available, tracing becomes a no-op.
    - If 'prometheus_client' is not available, metrics become no-op.
- Import and setup errors are caught and ignored so telemetry is optional and non-fatal.
- Console exporter and Prometheus HTTP server are convenient for local development.
"""

from __future__ import annotations
import contextlib
import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, start_http_server  # type: ignore
except Exception:  # optional
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    start_http_server = None  # type: ignore

try:
    from opentelemetry import trace  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import (  # type: ignore
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
                provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
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
    """Lightweight Prometheus metrics. No-op if prometheus_client missing.

    Extended for basic RAG observability: chunking, embeddings, vector DB queries.
    """

    def __init__(self, port: int = 8000):
        """Initialize Metrics."""
        self._enabled = Counter is not None and Gauge is not None
        if not self._enabled:
            return

        # Basic request/failure/latency metrics
        self.requests = Counter("ai_gateway_requests_total", "Total AI requests", ["provider"])
        self.failures = Counter("ai_gateway_failures_total", "Failed AI requests", ["provider"])
        self.latency = Gauge("ai_gateway_latency_seconds", "Request latency seconds", ["provider"])

        # Advanced RAG metrics: chunking, embeddings, vector DB
        self.chunks = Counter(
            "ai_gateway_chunks_total", "Total chunks produced during chunking", ["provider", "model"]
        )
        self.embedding_requests = Counter(
            "ai_gateway_embedding_requests_total", "Embedding requests total", ["provider", "model"]
        )
        self.embedding_latency = Gauge(
            "ai_gateway_embedding_latency_seconds", "Embedding latency seconds", ["provider", "model"]
        )
        self.vector_queries = Counter(
            "ai_gateway_vector_queries_total", "Vector DB queries total", ["provider", "model"]
        )
        self.vector_query_hits = Counter(
            "ai_gateway_vector_query_hits_total", "Vector DB query hits total", ["provider", "model"]
        )
        self.vector_query_latency = Gauge(
            "ai_gateway_vector_query_latency_seconds", "Vector DB query latency seconds", ["provider", "model"]
        )

        try:
            if start_http_server:
                start_http_server(port)
        except Exception as exc:
            logger.warning("Failed to start monitoring server: %s", exc, exc_info=True)

    # Basic metrics helpers
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

    # Advanced RAG helpers
    def record_chunks(self, provider: str, model: str, count: int = 1) -> None:
        """Record number of chunks produced during chunking/splitting."""
        if not self._enabled:
            return
        try:
            self.chunks.labels(provider=provider, model=model).inc(count)
        except Exception:
            logger.debug("Failed to record chunks metric", exc_info=True)

    def record_embedding(self, provider: str, model: str, seconds: float) -> None:
        """Record an embedding request and its latency."""
        if not self._enabled:
            return
        try:
            self.embedding_requests.labels(provider=provider, model=model).inc()
            self.embedding_latency.labels(provider=provider, model=model).set(seconds)
        except Exception:
            logger.debug("Failed to record embedding metrics", exc_info=True)

    def record_vector_query(self, provider: str, model: str, hits: int = 0, seconds: Optional[float] = None) -> None:
        """Record a vector DB / similarity search query.

        - hits: number of returned hits or candidates (if known).
        - seconds: query latency in seconds (optional).
        """
        if not self._enabled:
            return
        try:
            self.vector_queries.labels(provider=provider, model=model).inc()
            if hits:
                self.vector_query_hits.labels(provider=provider, model=model).inc(hits)
            if seconds is not None:
                self.vector_query_latency.labels(provider=provider, model=model).set(seconds)
        except Exception:
            logger.debug("Failed to record vector query metrics", exc_info=True)


@dataclass
class ModelQualityMetrics:
    """Collect simple quality metrics for a model and provider. Extended for RAG."""

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

    @property
    def avg_embedding_latency(self) -> float:
        """Average embedding latency per embedding request."""
        if self.total_embedding_requests == 0:
            return 0.0
        return self.total_embedding_latency_seconds / self.total_embedding_requests

    @property
    def avg_vector_query_latency(self) -> float:
        """Average vector query latency per query."""
        if self.total_vector_queries == 0:
            return 0.0
        return self.total_vector_query_latency_seconds / self.total_vector_queries


GLOBAL_TRACER = Tracer()
GLOBAL_METRICS = Metrics(port=int(os.getenv("PROMETHEUS_PORT", "8000")))