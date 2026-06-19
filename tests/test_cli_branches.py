def test_cli_missing_prompt_exit():
    from ai_cli import cli

    try:
        cli.main([])
    except SystemExit as e:
        assert e.code == 2

def test_cli_empty_prompt_exit():
    from ai_cli import cli
    try:
        cli.main(["--prompt", ""])
    except SystemExit as e:
        assert e.code == 2

def test_cli_valid_prompt(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr("ai_cli.cli.ask", lambda *a, **k: "ok")

    result = cli.main(["--prompt", "hello"])
    assert result in (0, None)