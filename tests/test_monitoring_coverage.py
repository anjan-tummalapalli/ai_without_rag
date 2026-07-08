from ai_cli.telemetry import monitoring  # isort:skip


def test_monitoring_basic_calls():
    # safe execution paths only
    if hasattr(monitoring, "init"):
        monitoring.init()

    if hasattr(monitoring, "track"):
        monitoring.track("test_event", {"x": 1})
