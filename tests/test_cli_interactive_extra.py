import builtins

import ai_cli.cli as cli


def test_run_interactive_help(monkeypatch, capsys):
    inputs = iter(["/help", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    assert cli.run_interactive("echo", None, 60) == 0
    out = capsys.readouterr().out
    assert "Commands:" in out


def test_run_interactive_switch_model_profile_stream(monkeypatch):
    inputs = iter([
        "/switch echo",
        "/model gpt-4o",
        "/profile dev",
        "/stream",
        "/quit",
    ])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    assert cli.run_interactive("auto", None, 60) == 0


def test_run_interactive_index_and_search(monkeypatch, tmp_path, capsys):
    f = tmp_path / "doc.txt"
    f.write_text("hello world", encoding="utf-8")

    inputs = iter([f"/index {f}", "/search hello", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(inputs))
    assert cli.run_interactive("echo", None, 60) == 0
    out = capsys.readouterr().out
    assert "RAG Context" in out or "Indexed file" in out
