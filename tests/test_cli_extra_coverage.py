import io

import pytest

import ai_cli.cli as cli


def test_main_rejects_missing_prompt(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    with pytest.raises(SystemExit):
        cli.main([])


def test_main_reads_prompt_from_stdin(monkeypatch):
    class FakeStdin:
        def isatty(self):
            return False

        @property
        def buffer(self):
            return io.BytesIO(b"hello from stdin")

    monkeypatch.setattr(cli.sys, "stdin", FakeStdin())
    monkeypatch.setattr(cli, "_invoke_with_retries", lambda kwargs: 0)

    assert cli.main([]) == 0


def test_main_debug_flag(monkeypatch):
    monkeypatch.setattr(cli, "_invoke_with_retries", lambda kwargs: 0)
    assert cli.main(["--debug", "-q", "hello"]) == 0


def test_main_truncates_large_prompt(monkeypatch):
    captured = {}

    def fake_invoke(kwargs):
        captured["prompt"] = kwargs["prompt"]
        return 0

    monkeypatch.setattr(cli, "_invoke_with_retries", fake_invoke)
    big = "x" * 100_001
    assert cli.main(["-q", big]) == 0
    assert len(captured["prompt"]) == 100_000


def test_invoke_with_retries_transient_error(monkeypatch):
    calls = {"n": 0}

    def fake_ask(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("temporary")
        return "ok"

    monkeypatch.setattr(cli, "ask", fake_ask)
    monkeypatch.setattr(cli.time, "sleep", lambda *_: None)

    assert cli._invoke_with_retries({"provider": "echo", "prompt": "hi"}) == 0
