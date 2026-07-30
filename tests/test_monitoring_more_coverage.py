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
