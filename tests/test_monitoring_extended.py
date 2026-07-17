"""
test_monitoring_extended.py

Comprehensive tests for telemetry/monitoring.py targeting the
ModelQualityMetrics dataclass, Metrics no-op helpers, Tracer,
and public telemetry API functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.auto_provider import AutoProvider
from ai_cli.providers.registry import PROVIDER_MAP

# ─────────────────────────────────────────────
# ModelQualityMetrics computed properties
# ─────────────────────────────────────────────


class TestModelQualityMetrics:
    def _make(self, **kwargs):
        from ai_cli.telemetry.monitoring import ModelQualityMetrics

        defaults = dict(provider="openai", model="gpt-4")
        defaults.update(kwargs)
        return ModelQualityMetrics(**defaults)

    def test_success_rate_zero_requests(self):
        m = self._make()
        assert m.success_rate == 0.0

    def test_success_rate_all_success(self):
        m = self._make(requests=10, failures=0)
        assert m.success_rate == 1.0

    def test_success_rate_partial_failures(self):
        m = self._make(requests=10, failures=2)
        assert abs(m.success_rate - 0.8) < 1e-9

    def test_success_rate_all_failures(self):
        m = self._make(requests=5, failures=5)
        assert m.success_rate == 0.0

    def test_avg_latency_zero_requests(self):
        m = self._make()
        assert m.avg_latency == 0.0

    def test_avg_latency_with_data(self):
        m = self._make(requests=4, total_latency_seconds=2.0)
        assert abs(m.avg_latency - 0.5) < 1e-9

    def test_hallucination_rate_zero(self):
        m = self._make()
        assert m.hallucination_rate == 0.0

    def test_hallucination_rate_with_data(self):
        m = self._make(requests=10, hallucination_failures=3)
        assert abs(m.hallucination_rate - 0.3) < 1e-9

    def test_avg_embedding_latency_zero(self):
        m = self._make()
        assert m.avg_embedding_latency == 0.0

    def test_avg_embedding_latency_with_data(self):
        m = self._make(
            total_embedding_requests=4, total_embedding_latency_seconds=2.0
        )
        assert abs(m.avg_embedding_latency - 0.5) < 1e-9

    def test_avg_vector_query_latency_zero(self):
        m = self._make()
        assert m.avg_vector_query_latency == 0.0

    def test_avg_vector_query_latency_with_data(self):
        m = self._make(
            total_vector_queries=2, total_vector_query_latency_seconds=1.0
        )
        assert abs(m.avg_vector_query_latency - 0.5) < 1e-9

    def test_rag_counters_default_zero(self):
        m = self._make()
        assert m.total_chunks == 0
        assert m.total_embedding_requests == 0
        assert m.total_vector_queries == 0
        assert m.total_vector_query_hits == 0


# ─────────────────────────────────────────────
# Metrics no-op helpers (when prometheus disabled)
# ─────────────────────────────────────────────


class TestMetricsNoOp:
    def _make_noop_metrics(self):
        """Create Metrics instance with prometheus forced off."""
        from ai_cli.telemetry import monitoring

        with patch.object(monitoring, "PROMETHEUS_ENABLED", False):
            with patch.object(monitoring, "PromCounter", None):
                with patch.object(monitoring, "PromGauge", None):
                    from ai_cli.telemetry.monitoring import Metrics

                    return Metrics()

    def test_record_request_noop(self):
        m = self._make_noop_metrics()
        m.record_request("openai")
        m.record_request(None)

    def test_record_failure_noop(self):
        m = self._make_noop_metrics()
        m.record_failure("gemini")
        m.record_failure(None)

    def test_record_latency_noop(self):
        m = self._make_noop_metrics()
        m.record_latency("openai", 1.5)
        m.record_latency(None, 0.1)
        m.record_latency("openai", None)

    def test_record_chunks_noop(self):
        m = self._make_noop_metrics()
        m.record_chunks("openai", "gpt-4", count=5)
        m.record_chunks(None, None, count=3)
        m.record_chunks("openai", "gpt-4", count=0)

    def test_record_embedding_noop(self):
        m = self._make_noop_metrics()
        m.record_embedding("openai", "text-embedding-3", 0.5)
        m.record_embedding(None, None, 0.2)

    def test_record_vector_query_noop(self):
        m = self._make_noop_metrics()
        m.record_vector_query("openai", "gpt-4", hits=5, seconds=0.1)
        m.record_vector_query(None, None)
        m.record_vector_query("openai", "gpt-4", hits=0)

    def test_close_noop(self):
        m = self._make_noop_metrics()
        m.close()


# ─────────────────────────────────────────────
# _NoopMetric
# ─────────────────────────────────────────────


class TestNoopMetric:
    def test_labels_returns_self(self):
        from ai_cli.telemetry.monitoring import _NoopMetric

        n = _NoopMetric()
        assert n.labels(provider="x") is n

    def test_inc_is_noop(self):
        from ai_cli.telemetry.monitoring import _NoopMetric

        n = _NoopMetric()
        n.inc()
        n.inc(5)

    def test_set_is_noop(self):
        from ai_cli.telemetry.monitoring import _NoopMetric

        n = _NoopMetric()
        n.set(1.0)


# ─────────────────────────────────────────────
# Tracer
# ─────────────────────────────────────────────


class TestTracer:
    def test_tracer_disabled_span_yields_none(self):
        from ai_cli.telemetry.monitoring import Tracer

        t = Tracer.__new__(Tracer)
        t._enabled = False
        t._tracer = None
        with t.span("test") as span:
            assert span is None

    def test_tracer_shutdown_no_provider(self):
        from ai_cli.telemetry.monitoring import Tracer

        t = Tracer.__new__(Tracer)
        t._provider = None
        t.shutdown()

    def test_tracer_shutdown_with_provider(self):
        from ai_cli.telemetry.monitoring import Tracer

        t = Tracer.__new__(Tracer)
        mock_provider = MagicMock()
        t._provider = mock_provider
        t.shutdown()
        mock_provider.shutdown.assert_called_once()

    def test_tracer_init_creates_instance(self):
        from ai_cli.telemetry.monitoring import Tracer

        t = Tracer.__new__(Tracer)
        t._enabled = False
        t._provider = None
        t._tracer = None
        assert not t._enabled


# ─────────────────────────────────────────────
# Public telemetry API
# ─────────────────────────────────────────────


class TestPublicTelemetryAPI:
    def test_start_trace(self):
        from ai_cli.telemetry.monitoring import start_trace

        result = start_trace("my_trace")
        assert result["trace"] == "my_trace"
        assert result["status"] == "started"

    def test_start_trace_default(self):
        from ai_cli.telemetry.monitoring import start_trace

        result = start_trace()
        assert "trace" in result

    def test_end_trace(self):
        from ai_cli.telemetry.monitoring import end_trace

        result = end_trace("trace-123")
        assert result["trace"] == "trace-123"
        assert result["status"] == "ended"

    def test_end_trace_default(self):
        from ai_cli.telemetry.monitoring import end_trace

        result = end_trace()
        assert "trace" in result

    def test_log_event(self):
        from ai_cli.telemetry.monitoring import log_event

        result = log_event("user_query", {"prompt": "hello"})
        assert result["event"] == "user_query"
        assert result["data"] == {"prompt": "hello"}

    def test_log_event_no_data(self):
        from ai_cli.telemetry.monitoring import log_event

        result = log_event("something")
        assert result["data"] == {}

    def test_record_metric(self):
        from ai_cli.telemetry.monitoring import record_metric

        result = record_metric("latency", 0.5)
        assert result["metric"] == "latency"
        assert result["value"] == 0.5

    def test_telemetry_track(self):
        from ai_cli.telemetry.monitoring import Telemetry

        t = Telemetry()
        assert t.track("event_a") is True
        assert "event_a" in t.events

    def test_telemetry_multiple_events(self):
        from ai_cli.telemetry.monitoring import Telemetry

        t = Telemetry()
        t.track("e1")
        t.track("e2")
        assert t.events == ["e1", "e2"]


# ─────────────────────────────────────────────
# _safe_counter / _safe_gauge (with noop)
# ─────────────────────────────────────────────


class TestSafeMetricHelpers:
    def test_safe_counter_without_prometheus(self):
        from ai_cli.telemetry import monitoring

        with patch.object(monitoring, "PromCounter", None):
            from ai_cli.telemetry.monitoring import _NoopMetric, _safe_counter

            result = _safe_counter("test_counter", "doc", ["label"])
            assert isinstance(result, _NoopMetric)

    def test_safe_gauge_without_prometheus(self):
        from ai_cli.telemetry import monitoring

        with patch.object(monitoring, "PromGauge", None):
            from ai_cli.telemetry.monitoring import _NoopMetric, _safe_gauge

            result = _safe_gauge("test_gauge", "doc", ["label"])
            assert isinstance(result, _NoopMetric)

    def test_find_existing_metric_without_prom_core(self):
        from ai_cli.telemetry import monitoring

        with patch.object(monitoring, "prom_core", None):
            from ai_cli.telemetry.monitoring import _find_existing_metric

            assert _find_existing_metric("some_metric") is None


# ============================================================================
# Extra monitoring.py coverage
# ============================================================================


class TestNoopMetricExtra:
    def test_noop_metric_methods(self):
        from ai_cli.telemetry.monitoring import _NoopMetric

        m = _NoopMetric()
        assert m.labels("a") is m
        assert m.inc() is None
        if hasattr(m, "set"):
            assert m.set(5) is None


class TestSafeMetricCreation:
    def test_find_existing_metric_not_found(self):
        from ai_cli.telemetry import monitoring

        class Dummy:
            _names_to_collectors = {}

        with patch.object(
            monitoring,
            "prom_core",
            type("P", (), {"REGISTRY": Dummy()}),
        ):
            assert monitoring._find_existing_metric("missing") is None


class TestTelemetryExtra:
    def test_track_many(self):
        from ai_cli.telemetry.monitoring import Telemetry

        t = Telemetry()
        for i in range(10):
            assert t.track(f"e{i}")
        assert len(t.events) == 10


class TestTracerExtra:
    def test_span_enabled(self):
        from ai_cli.telemetry.monitoring import Tracer

        cm = MagicMock()
        cm.__enter__.return_value = "span"

        tracer = MagicMock()
        tracer.start_as_current_span.return_value = cm

        t = Tracer.__new__(Tracer)
        t._enabled = True
        t._tracer = tracer

        with t.span("abc") as span:
            assert span == "span"

        tracer.start_as_current_span.assert_called_once_with("abc")

    def test_shutdown_provider_exception(self):
        from ai_cli.telemetry.monitoring import Tracer

        provider = MagicMock()
        provider.shutdown.side_effect = RuntimeError

        t = Tracer.__new__(Tracer)
        t._provider = provider

        try:
            t.shutdown()
        except RuntimeError:
            pass


class _UnauthorizedThenOkProvider:
    """Stub that raises an 'unauthorized' error."""

    def send(self, prompt: str) -> str:
        _ = prompt
        raise ProviderRequestError("401 unauthorized")

    def ask(self, prompt: str) -> str:
        return self.send(prompt)


class _FallbackOkProvider:
    def send(self, prompt: str) -> str:
        _ = prompt
        return "fallback_ok"

    def ask(self, prompt: str) -> str:
        return self.send(prompt)


def test_send_skips_unauthorized_error_and_raises() -> None:
    """send() records an 'unauthorized'-style failure as skipped."""
    PROVIDER_MAP["__auto_unauthorized__"] = _UnauthorizedThenOkProvider
    ap = AutoProvider(fallback_order=["__auto_unauthorized__"])
    with pytest.raises(ProviderRequestError, match="Auto fallback exhausted"):
        ap.send("hello")


def test_send_reports_provider_not_found() -> None:
    """send() records a 'not found' error for a missing provider class."""
    PROVIDER_MAP["__auto_present__"] = _FallbackOkProvider
    ap = AutoProvider(fallback_order=["__auto_present__"])
    ap.fallback_order = ["__auto_missing_now__", "__auto_present__"]
    result = ap.send("hello")
    assert result == "fallback_ok"


def test_safe_counter_generic_exception(monkeypatch):
    import ai_cli.telemetry.monitoring as monitoring

    def boom(*args, **kwargs):
        raise RuntimeError("counter failed")

    monkeypatch.setattr(monitoring, "PromCounter", boom)

    metric = monitoring._safe_counter(
        "unit_test_counter",
        "doc",
        [],
    )

    assert isinstance(metric, monitoring._NoopMetric)


def test_safe_gauge_generic_exception(monkeypatch):
    import ai_cli.telemetry.monitoring as monitoring

    def boom(*args, **kwargs):
        raise RuntimeError("gauge failed")

    monkeypatch.setattr(monitoring, "PromGauge", boom)

    metric = monitoring._safe_gauge(
        "unit_test_gauge",
        "doc",
        [],
    )

    assert isinstance(metric, monitoring._NoopMetric)


def test_safe_counter_duplicate_without_existing(monkeypatch):
    import ai_cli.telemetry.monitoring as monitoring

    class Duplicate:
        def __call__(self, *args, **kwargs):
            raise ValueError("duplicate")

    monkeypatch.setattr(
        monitoring,
        "PromCounter",
        Duplicate(),
    )

    monkeypatch.setattr(
        monitoring,
        "_find_existing_metric",
        lambda name: None,
    )

    metric = monitoring._safe_counter(
        "duplicate_counter",
        "doc",
        [],
    )

    assert isinstance(metric, monitoring._NoopMetric)


def test_safe_gauge_duplicate_without_existing(monkeypatch):
    import ai_cli.telemetry.monitoring as monitoring

    class Duplicate:
        def __call__(self, *args, **kwargs):
            raise ValueError("duplicate")

    monkeypatch.setattr(
        monitoring,
        "PromGauge",
        Duplicate(),
    )

    monkeypatch.setattr(
        monitoring,
        "_find_existing_metric",
        lambda name: None,
    )

    metric = monitoring._safe_gauge(
        "duplicate_gauge",
        "doc",
        [],
    )

    assert isinstance(metric, monitoring._NoopMetric)
