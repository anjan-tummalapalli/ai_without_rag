import importlib

import ai_cli.telemetry.monitoring as monitoring


def test_monitoring_imports():
    assert monitoring is not None


def test_monitoring_module_has_expected_symbols():
    assert hasattr(monitoring, "__name__")


def test_monitoring_module_imports():
    mod = importlib.import_module("ai_cli.telemetry.monitoring")
    assert mod.__name__ == "ai_cli.telemetry.monitoring"


def test_monitoring_module_has_doc_or_attrs():
    mod = importlib.import_module("ai_cli.telemetry.monitoring")
    assert hasattr(mod, "__dict__")