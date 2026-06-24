import io
from unittest.mock import MagicMock

import pytest

from ai_cli import cli
from ai_cli.ai_chat import chunk_text
from ai_cli.cli import (
    _decode_chunk,
    _safe_resolve_path,
    _sanitize_log_value,
    build_parser,
)


def test_cli_build_parser():
    parser = build_parser()

    args = parser.parse_args(
        ["--prompt", "hello"]
    )

    assert args.prompt == "hello"


def test_cli_help():
    parser = build_parser()

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])

    assert exc.value.code == 0


def test_sanitize_log_value():
    assert _sanitize_log_value(None) == "None"
    assert _sanitize_log_value("hello") == "hello"


def test_safe_resolve_path_empty():
    assert _safe_resolve_path("") is not None


def test_decode_chunk_string():
    assert _decode_chunk("hello") == "hello"


def test_decode_chunk_bytes():
    assert _decode_chunk(b"hello") == "hello"


def test_missing_prompt(monkeypatch):
    monkeypatch.setattr(
        "ai_cli.cli._read_stdin_prompt",
        lambda: "",
    )

    with pytest.raises(SystemExit) as exc:
        cli.main([])

    assert exc.value.code == 2


def test_empty_prompt(monkeypatch):
    monkeypatch.setattr(
        "ai_cli.cli._read_stdin_prompt",
        lambda: "",
    )

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--prompt",
                "",
            ]
        )

    assert exc.value.code == 2

def test_read_stdin_prompt(monkeypatch):
    monkeypatch.setattr(
        "sys.stdin",
        io.TextIOWrapper(
            io.BytesIO(b"hello from stdin")
        ),
    )

    assert cli._read_stdin_prompt() == "hello from stdin"

def test_timeout_validation():
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--prompt",
                "hello",
                "--timeout",
                "0",
            ]
        )

    assert exc.value.code == 2

def test_large_prompt_truncation(monkeypatch):
    monkeypatch.setattr(
        "ai_cli.cli._invoke_with_retries",
        lambda kwargs: 0,
    )

    result = cli.main(
        [
            "--prompt",
            "x" * 100001,
        ]
    )

    assert result == 0

def test_load_rag_docs_text():
    pipeline = cli._load_rag_docs(
        ["hello world"],
        500,
        50,
    )

    assert pipeline is not None

def test_interactive_exit(monkeypatch):
    monkeypatch.setattr(
        "builtins.input",
        lambda _: "/exit",
    )

    result = cli.run_interactive(
        provider="auto",
        model=None,
        timeout=30,
    )

    assert result == 0

def test_safe_resolve_path_none():
    assert cli._safe_resolve_path(None) is None

def test_chunk_validation():
    with pytest.raises(ValueError):
        chunk_text(
            "abc",
            chunk_size=0
        )

def test_cli_no_args_exit(monkeypatch):
    import pytest

    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli._read_stdin_prompt",
        lambda: (_ for _ in ()).throw(SystemExit())
    )

    with pytest.raises(SystemExit):
        cli.main([])

def test_cli_invalid_provider():
    with pytest.raises(SystemExit):
        cli.main([
            "--provider",
            "invalid-provider",
            "hello"
        ])

def test_cli_provider_error(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli.ask",
        lambda *a, **k: (_ for _ in ()).throw(
            Exception("bad provider")
        )
    )

    with pytest.raises(Exception, match="bad provider"):
        cli.main([
            "--provider",
            "bad",
            "--prompt",
            "hello",
        ])

def test_cli_stdin_prompt(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli.init_providers",
        lambda: None
    )

    monkeypatch.setattr(
        "ai_cli.providers.registry.build_provider",
        lambda *a, **k: MagicMock(
            send=lambda x: "hello"
        )
    )

    monkeypatch.setattr(
        "ai_cli.cli._read_stdin_prompt",
        lambda: "hello"
    )

    cli.main([])

def test_cli_prompt_success(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli.init_providers",
        lambda: None
    )

    monkeypatch.setattr(
        "ai_cli.cli.ask",
        lambda **kwargs: "hello"
    )

    result = cli.main(
        [
            "--prompt",
            "hello",
            "--provider",
            "auto",
        ]
    )

    assert result == 0


def test_cli_stream_mode(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli.init_providers",
        lambda: None
    )

    monkeypatch.setattr(
        "ai_cli.cli.ask",
        lambda **kwargs: "stream ok"
    )

    result = cli.main(
        [
            "--prompt",
            "hello",
            "--stream",
        ]
    )

    assert result == 0


def test_cli_debug(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli.init_providers",
        lambda: None
    )

    monkeypatch.setattr(
        "ai_cli.cli.ask",
        lambda **kwargs: "ok"
    )

    assert cli.main(
        [
            "--prompt",
            "debug",
            "--debug",
        ]
    ) == 0

def test_cli_timeout_success(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli.init_providers",
        lambda: None
    )

    monkeypatch.setattr(
        "ai_cli.cli.ask",
        lambda **kwargs: "ok"
    )

    assert cli.main(
        [
            "--prompt",
            "hello",
            "--timeout",
            "20",
        ]
    ) == 0

def test_cli_debug_success(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        "ai_cli.cli.init_providers",
        lambda: None
    )

    monkeypatch.setattr(
        "ai_cli.cli.ask",
        lambda **kwargs: "ok"
    )

    assert cli.main(
        [
            "--prompt",
            "hello",
            "--debug",
        ]
    ) == 0

def test_run_interactive_exit(monkeypatch):
    from ai_cli.cli import run_interactive

    inputs = iter(["/exit"])

    monkeypatch.setattr(
        "builtins.input",
        lambda *_: next(inputs)
    )

    assert run_interactive(
        provider="openai",
        model=None,
        timeout=30,
    ) == 0

def test_run_interactive_commands(monkeypatch):
    from ai_cli.cli import run_interactive

    inputs = iter([
        "/help",
        "/switch gemini",
        "/model test-model",
        "/profile dev",
        "/stream",
        "/exit",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda *_: next(inputs)
    )

    assert run_interactive(
        provider="openai",
        model=None,
        timeout=30,
    ) == 0

def test_run_interactive_prompt(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        cli,
        "_invoke_with_retries",
        lambda kwargs: 0
    )

    inputs = iter([
        "hello",
        "/exit",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda *_: next(inputs)
    )

    assert cli.run_interactive(
        provider="openai",
        model=None,
        timeout=30,
    ) == 0

def test_cli_reads_stdin(monkeypatch):
    import io
    import sys

    from ai_cli import cli

    class FakeStdin:
        def __init__(self):
            self.buffer = io.BytesIO(b"hello")

        def isatty(self):
            return False

    monkeypatch.setattr(
        sys,
        "stdin",
        FakeStdin()
    )

    monkeypatch.setattr(
        cli,
        "_invoke_with_retries",
        lambda kwargs: 0
    )

    assert cli.main([]) == 0

def test_sync_result_dict():
    from ai_cli.cli import _handle_sync_result

    assert _handle_sync_result(
        {"a":1}
    ) == 0

def test_sync_result_iterable():
    from ai_cli.cli import _handle_sync_result

    assert _handle_sync_result(
        ["a","b"]
    ) == 0

def test_decode_chunk():
    from ai_cli.cli import _decode_chunk

    assert _decode_chunk(b"hello") == "hello"
    assert _decode_chunk("x") == "x"

def test_interactive_help_exit(monkeypatch):
    from ai_cli import cli

    inputs = iter([
        "/help",
        "/exit",
    ])

    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    assert (
        cli.run_interactive(
            provider="echo",
            model=None,
            timeout=60,
        )
        == 0
    )

def test_cli_rag_prompt(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        cli,
        "ask",
        lambda **kwargs: "ok"
    )

    assert cli.main([
        "--prompt",
        "hello",
        "--rag"
    ]) == 0

def test_interactive_commands(monkeypatch):
    from ai_cli import cli

    commands = iter([
        "/switch gemini",
        "/model test-model",
        "/profile dev",
        "/stream",
        "/exit",
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(commands)
    )

    assert cli.run_interactive(
        provider="echo",
        model=None,
        timeout=60,
    ) == 0

def test_read_stdin_prompt_bytes(monkeypatch):
    import io

    from ai_cli import cli

    class FakeStdin:
        buffer = io.BytesIO(b"hello from stdin")

    monkeypatch.setattr(cli, "sys", type("X", (), {"stdin": FakeStdin()})())

    assert cli._read_stdin_prompt() == "hello from stdin"

def test_timeout_negative(monkeypatch):
    from ai_cli import cli

    with pytest.raises(SystemExit):
        cli.main(["-q", "hello", "--timeout", "-1"])

def test_cli_unknown_provider(monkeypatch, caplog):
    from ai_cli import cli

    cli.main([
        "--provider",
        "does-not-exist",
        "-q",
        "hello",
    ])

    assert "build_provider" in caplog.text

def test_auto_provider_init_failure(monkeypatch):
    from ai_cli.providers import auto_provider
    from ai_cli.providers.auto_provider import AutoProvider

    class Bad:
        def __init__(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        auto_provider,
        "PROVIDER_MAP",
        {"bad": Bad},
    )

    p = AutoProvider(
        fallback_order=["bad"]
    )

    with pytest.raises(Exception):
        p.send("x")

def test_deepseek_health_check_without_key(monkeypatch):
    from ai_cli.providers.deepseek_provider import DeepSeekProvider

    monkeypatch.delenv(
        "DEEPSEEK_API_KEY",
        raising=False,
    )

    provider = DeepSeekProvider()

    assert provider.health_check() is False

def test_prompt_truncation_in_main(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        cli,
        "_MAX_PROMPT_BYTES",
        10,
        raising=False,
    )

    monkeypatch.setattr(
        cli,
        "ask",
        lambda **kwargs: "ok",
    )

    result = cli.main([
        "-q",
        "x" * 100,
    ])

    assert result == 0