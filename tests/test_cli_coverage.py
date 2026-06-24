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
    import io

    from ai_cli import cli

    monkeypatch.setattr(
        "sys.stdin",
        io.TextIOWrapper(
            io.BytesIO(b"hello from stdin")
        ),
    )

    assert cli._read_stdin_prompt() == "hello from stdin"

def test_timeout_validation():
    from ai_cli import cli

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
    from ai_cli import cli

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
    from ai_cli import cli

    pipeline = cli._load_rag_docs(
        ["hello world"],
        500,
        50,
    )

    assert pipeline is not None

def test_interactive_exit(monkeypatch):
    from ai_cli import cli

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