from unittest.mock import MagicMock, patch

import pytest

from ai_cli import cli
from ai_cli.cli import run_interactive
from ai_cli.core.resilience import RetryEngine


def test_retry_engine_exception_retry():
    engine = RetryEngine(
        max_attempts=2
    )

    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        engine.execute(fail)

    assert len(calls) == 2


def test_retry_engine_retry_on_tuple_break():
    engine = RetryEngine(
        max_attempts=3,
        retry_on=(ValueError,)
    )

    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        engine.execute(fail)

    # RuntimeError is not retryable, should stop immediately
    assert len(calls) == 1


def test_retry_engine_retry_filter_break():
    engine = RetryEngine(
        max_attempts=3,
        retry_on=lambda exc: False
    )

    calls = []

    def fail():
        calls.append(1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        engine.execute(fail)

    assert len(calls) == 1


def test_retry_engine_base_delay():
    engine = RetryEngine(
        max_attempts=2,
        base_delay=0.01
    )

    with patch(
        "ai_cli.core.resilience.time.sleep"
    ) as sleep:

        def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            engine.execute(fail)

        sleep.assert_called()


def test_interactive_profile_and_stream():
    inputs = iter([
        "/profile test",
        "/stream",
        "/profile",
        "exit",
    ])

    with patch("builtins.input", side_effect=inputs):
        run_interactive(
            provider="deepseek",
            model=None,
            timeout=60,
        )

def test_interactive_index_text():

    pipeline = MagicMock()

    inputs = iter([
        "/index hello world",
        "exit",
    ])

    with patch(
        "builtins.input",
        side_effect=inputs
    ):
        run_interactive(
            provider="deepseek",
            model=None,
            timeout=60,
            rag=pipeline,
        )

    pipeline.upsert_documents.assert_called()

def test_cli_stdin_empty(monkeypatch):

    monkeypatch.setattr(
        "sys.stdin.isatty",
        lambda: False
    )

    monkeypatch.setattr(
        "sys.stdin.buffer.read",
        lambda x: b""
    )

    with pytest.raises(SystemExit):
        cli.main([])

def test_interactive_profile_stream():
    inputs = iter([
        "/profile test-profile",
        "/stream",
        "/profile",
        "exit",
    ])

    with patch("builtins.input", side_effect=inputs):
        run_interactive(
            provider="deepseek",
            model=None,
            timeout=60,
        )

def test_interactive_search():
    pipeline = MagicMock()
    pipeline.retrieve_context.return_value = "k8s context"

    inputs = iter([
        "/search kubernetes",
        "/search",
        "exit",
    ])

    with patch(
        "builtins.input",
        side_effect=inputs
    ):
        run_interactive(
            provider="deepseek",
            model=None,
            timeout=60,
            rag=pipeline,
        )

    pipeline.retrieve_context.assert_called()

def test_interactive_index_raw_text():
    pipeline = MagicMock()

    inputs = iter([
        "/index hello kubernetes",
        "exit",
    ])

    with patch(
        "builtins.input",
        side_effect=inputs
    ):
        run_interactive(
            provider="deepseek",
            model=None,
            timeout=60,
            rag=pipeline,
        )

    pipeline.upsert_documents.assert_called()