import importlib
from types import SimpleNamespace

import ai_cli.telemetry.monitoring as monitoring
import pytest

MODULE_PATH = "ai_cli.telemetry.monitoring"


def test_monitoring_module_imports() -> None:
    """Telemetry module imports successfully."""
    module = importlib.import_module(MODULE_PATH)
    assert module is not None


def test_monitoring_has_expected_attributes() -> None:
    """Telemetry module exposes at least one public API."""
    module = importlib.import_module(MODULE_PATH)

    possible_attrs = (
        "start_trace",
        "end_trace",
        "log_event",
        "record_metric",
        "Telemetry",
        "monitor",
        "track",
        "Metrics",
        "Tracer",
    )

    assert any(hasattr(module, attr) for attr in possible_attrs)


def test_monitoring_functions_execute_safely() -> None:
    """Public callables should not unexpectedly crash."""
    module = importlib.import_module(MODULE_PATH)

    callables = [
        getattr(module, name)
        for name in dir(module)
        if callable(getattr(module, name)) and not name.startswith("_")
    ]

    if not callables:
        pytest.skip("No callable telemetry functions found")

    for func in callables:
        try:
            func()
        except TypeError:
            try:
                func(event="test_event")
            except TypeError:
                try:
                    func("test_event")
                except Exception:
                    pass
        except Exception as exc:
            pytest.fail(f"{func} raised unexpected exception: {exc}")


def test_monitoring_module_is_import_safe_multiple_times() -> None:
    """Repeated imports should return same module."""
    first = importlib.import_module(MODULE_PATH)
    second = importlib.import_module(MODULE_PATH)

    assert first is second


def test_find_existing_metric_without_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyCore:
        REGISTRY = None

    monkeypatch.setattr(monitoring, "prom_core", DummyCore())

    assert monitoring._find_existing_metric("metric") is None


def test_find_existing_metric_collectors_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collector = SimpleNamespace(_names={"metric_one"})

    registry = SimpleNamespace(
        _collector_to_names=None,
        collectors=[collector],
    )

    monkeypatch.setattr(
        monitoring,
        "prom_core",
        SimpleNamespace(REGISTRY=registry),
    )

    assert monitoring._find_existing_metric("metric_one") is collector


def test_find_existing_metric_collectors_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Registry:
        @property
        def collectors(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        monitoring,
        "prom_core",
        SimpleNamespace(REGISTRY=Registry()),
    )

    assert monitoring._find_existing_metric("metric") is None


def test_tracer_init_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadProvider:
        def __init__(self) -> None:
            raise RuntimeError()

    monkeypatch.setattr(monitoring, "trace", object())
    monkeypatch.setattr(monitoring, "SDKTracerProvider", BadProvider)
    monkeypatch.setattr(monitoring, "SimpleSpanProcessor", object)
    monkeypatch.setattr(monitoring, "ConsoleSpanExporter", object)

    tracer = monitoring.Tracer()

    assert tracer._enabled is False


def test_tracer_shutdown_called() -> None:
    called = {}

    class Provider:
        def shutdown(self) -> None:
            called["shutdown"] = True

    tracer = monitoring.Tracer()
    tracer._provider = Provider()

    tracer.shutdown()

    assert called["shutdown"]


def test_metrics_server_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_server(*args, **kwargs):
        calls.append((args, kwargs))
        if kwargs:
            raise TypeError()
        return None

    monkeypatch.setattr(
        monitoring,
        "start_http_server",
        fake_server,
    )

    monitoring.Metrics()

    assert len(calls) == 2


def test_metrics_server_fallback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_server(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise TypeError()
        raise RuntimeError()

    monkeypatch.setattr(
        monitoring,
        "start_http_server",
        fake_server,
    )

    monitoring.Metrics()

    assert len(calls) == 2


def test_record_embedding_without_latency() -> None:
    metrics = monitoring.Metrics()

    metrics.record_embedding(
        "provider",
        "model",
        None,
    )


def test_record_request_metric_failure() -> None:
    metrics = monitoring.Metrics()

    class Broken:
        def labels(self, **kwargs):
            raise RuntimeError()

    metrics.requests = Broken()

    metrics.record_request("openai")
