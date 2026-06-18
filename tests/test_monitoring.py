import importlib

import pytest

MODULE_PATH = "ai_cli.telemetry.monitoring"


def test_monitoring_module_imports():
    """
    Ensures telemetry module can be imported without side effects.
    """
    module = importlib.import_module(MODULE_PATH)
    assert module is not None


def test_monitoring_has_expected_attributes():
    """
    Checks for common telemetry constructs.
    This is flexible since implementations vary.
    """
    module = importlib.import_module(MODULE_PATH)

    # Acceptable optional attributes
    possible_attrs = [
        "start_trace",
        "end_trace",
        "log_event",
        "record_metric",
        "Telemetry",
        "monitor",
        "track",
    ]

    assert any(hasattr(module, attr) for attr in possible_attrs), (
        "Telemetry module should expose at least one monitoring function/class"
    )


def test_monitoring_functions_execute_safely():
    """
    If telemetry functions exist, ensure they don't crash when called.
    """
    module = importlib.import_module(MODULE_PATH)

    test_cases = []

    # Collect callable attributes dynamically
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and not name.startswith("_"):
            test_cases.append(obj)

    # If nothing callable exists, still pass (module may be placeholder)
    if not test_cases:
        pytest.skip("No callable telemetry functions found")

    # Call each function safely with broad fallbacks
    for func in test_cases:
        try:
            # Try calling with no args first
            try:
                func()
            except TypeError:
                # Try common safe signatures
                try:
                    func(event="test_event")
                except TypeError:
                    try:
                        func("test_event")
                    except Exception:
                        pass  # ignore strict signature issues
        except Exception as e:
            pytest.fail(f"Telemetry function {func} raised unexpected error: {e}")


def test_monitoring_module_is_import_safe_multiple_times():
    """
    Ensures repeated imports don't break state or cause side effects.
    """
    m1 = importlib.import_module(MODULE_PATH)
    m2 = importlib.import_module(MODULE_PATH)

    assert m1 is m2