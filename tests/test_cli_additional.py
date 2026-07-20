import builtins
from unittest.mock import MagicMock

import ai_cli.cli as cli


def test_make_service():
    svc = cli._make_service(
        provider="openai",
        model="gpt-4",
        timeout=15,
        profile="dev",
        stream=True,
        modules="a,b",
    )

    assert svc.provider == "openai"
    assert svc.model == "gpt-4"
    assert svc.timeout == 15
    assert svc.profile == "dev"
    assert svc.stream is True


def test_call_service_timeout(capsys):
    svc = MagicMock()
    svc.ask.side_effect = TimeoutError("boom")

    assert cli._call_service(svc, "hello") == 124
    assert "boom" in capsys.readouterr().err


def test_call_service_connection(capsys):
    svc = MagicMock()
    svc.ask.side_effect = ConnectionError("offline")

    assert cli._call_service(svc, "hello") == 1
    assert "offline" in capsys.readouterr().err


def test_call_service_runtime(capsys):
    svc = MagicMock()
    svc.ask.side_effect = RuntimeError("failed")

    assert cli._call_service(svc, "hello") == 1
    assert "failed" in capsys.readouterr().err


def test_load_rag_docs_bad_path(monkeypatch):
    monkeypatch.setattr(cli.os.path, "exists", lambda _: True)
    monkeypatch.setattr(cli, "_safe_resolve_path", lambda _: None)

    pipeline = cli._load_rag_docs(["abc"], 100, 10)

    assert pipeline is not None


def test_load_rag_docs_open_failure(monkeypatch):
    monkeypatch.setattr(cli.os.path, "exists", lambda _: True)
    monkeypatch.setattr(cli, "_safe_resolve_path", lambda _: "/tmp/x")

    def bad_open(*args, **kwargs):
        raise OSError("cannot open")

    monkeypatch.setattr(builtins, "open", bad_open)

    pipeline = cli._load_rag_docs(["abc"], 100, 10)

    assert pipeline is not None


def test_main_interactive(monkeypatch):
    monkeypatch.setattr(cli, "init_providers", lambda: None)
    monkeypatch.setattr(cli, "run_interactive", lambda *a, **k: 99)

    rc = cli.main(["--interactive"])

    assert rc == 99


def test_main_rag_context(monkeypatch):
    class DummyPipeline:
        def retrieve_context(self, prompt, top_k):
            return "context"

    monkeypatch.setattr(cli, "init_providers", lambda: None)
    monkeypatch.setattr(cli, "_load_rag_docs", lambda *a, **k: DummyPipeline())

    captured = {}

    def fake_invoke(kwargs):
        captured["prompt"] = kwargs["prompt"]
        return 0

    monkeypatch.setattr(cli, "_invoke_with_retries", fake_invoke)

    assert cli.main(["--rag", "-q", "hello"]) == 0
    assert "Context:" in captured["prompt"]


def test_run_interactive_bad_path(monkeypatch, capsys):
    class DummyPipeline:
        def retrieve_context(self, *a, **k):
            return ""

        def upsert_documents(self, *a, **k):
            pass

    inputs = iter(
        [
            "/index file.txt",
            "/exit",
        ]
    )

    monkeypatch.setattr(cli.os.path, "exists", lambda _: True)
    monkeypatch.setattr(cli, "_safe_resolve_path", lambda _: None)
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))

    assert (
        cli.run_interactive(
            "auto",
            None,
            30,
            rag=DummyPipeline(),
        )
        == 0
    )

    assert "illegal traversal" in capsys.readouterr().err


def test_run_interactive_command_failure(monkeypatch, capsys):
    class DummyPipeline:
        def retrieve_context(self, *a, **k):
            return ""

    inputs = iter(
        [
            "hello",
            "/exit",
        ]
    )

    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    monkeypatch.setattr(cli, "_invoke_with_retries", lambda kwargs: 5)

    assert (
        cli.run_interactive(
            "auto",
            None,
            30,
            rag=DummyPipeline(),
        )
        == 0
    )

    assert "command failed" in capsys.readouterr().err
