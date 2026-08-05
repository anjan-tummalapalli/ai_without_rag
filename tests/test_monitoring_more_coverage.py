import ai_cli.telemetry.monitoring as monitoring


def test_find_existing_metric_using_collectors(monkeypatch):
    class Collector:
        _names = {"metric_a"}

    class Registry:
        collectors = [Collector()]

    class Core:
        REGISTRY = Registry()

    monkeypatch.setattr(monitoring, "prom_core", Core)

    assert (
        monitoring._find_existing_metric("metric_a") is Registry.collectors[0]
    )


def test_find_existing_metric_handles_exception(monkeypatch):
    class Registry:
        @property
        def collectors(self):
            raise RuntimeError("boom")

    class Core:
        REGISTRY = Registry()

    monkeypatch.setattr(monitoring, "prom_core", Core)

    assert monitoring._find_existing_metric("metric") is None


def test_monitoring_http_server_success(monkeypatch):
    started = {}

    def fake_server(port, addr=None):
        started["port"] = port
        started["addr"] = addr

    monkeypatch.setattr(monitoring, "start_http_server", fake_server)

    monitoring.Monitoring(host="127.0.0.1", port=9000)

    assert started == {
        "port": 9000,
        "addr": "127.0.0.1",
    }


def test_monitoring_http_server_typeerror_fallback(monkeypatch):
    calls = []

    def fake_server(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise TypeError
        return None

    monkeypatch.setattr(monitoring, "start_http_server", fake_server)

    monitoring.Monitoring(port=9100)

    assert len(calls) == 2
    assert calls[1][0] == (9100,)


def test_monitoring_http_server_general_exception(monkeypatch):
    def fake_server(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(monitoring, "start_http_server", fake_server)

    # Should not raise
    monitoring.Monitoring(port=9200)


def test_noop_metric():
    m = monitoring._NoopMetric()

    assert m.labels("a") is m
    assert m.inc() is None
    assert m.set() is None


def test_find_existing_metric_without_prom_core(monkeypatch):
    monkeypatch.setattr(monitoring, "prom_core", None)

    assert monitoring._find_existing_metric("metric") is None


def test_safe_counter_without_prometheus(monkeypatch):
    monkeypatch.setattr(monitoring, "PromCounter", None)

    metric = monitoring._safe_counter(
        "counter",
        "doc",
        [],
    )

    assert isinstance(metric, monitoring._NoopMetric)


def test_safe_gauge_without_prometheus(monkeypatch):
    monkeypatch.setattr(monitoring, "PromGauge", None)

    metric = monitoring._safe_gauge(
        "gauge",
        "doc",
        [],
    )

    assert isinstance(metric, monitoring._NoopMetric)


def test_find_existing_metric_registry_none(monkeypatch):
    class Core:
        REGISTRY = None

    monkeypatch.setattr(monitoring, "prom_core", Core)

    assert monitoring._find_existing_metric("metric") is None


def test_monitoring_logger_exception(monkeypatch):
    def fake_server(*args, **kwargs):
        raise RuntimeError("boom")

    messages = []

    class FakeLogger:
        def exception(self, msg, *args, **kwargs):
            messages.append(msg)

        def info(self, *args, **kwargs):
            pass

    monkeypatch.setattr(monitoring, "logger", FakeLogger())
    monkeypatch.setattr(monitoring, "start_http_server", fake_server)

    monitoring.Monitoring(port=9876)

    assert messages == ["Failed to start monitoring server"]


def test_noop_metric_all_methods():
    metric = monitoring._NoopMetric()

    assert metric.labels("a", "b") is metric
    assert metric.labels() is metric
    assert metric.inc() is None
    assert metric.inc(5) is None
    assert metric.set() is None
    assert metric.set(10) is None


def test_find_existing_metric_no_names(monkeypatch):
    class Collector:
        pass

    class Registry:
        collectors = [Collector()]

    class Core:
        REGISTRY = Registry()

    monkeypatch.setattr(monitoring, "prom_core", Core)

    assert monitoring._find_existing_metric("metric") is None
