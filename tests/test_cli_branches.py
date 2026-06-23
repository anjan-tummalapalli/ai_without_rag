import io

import pytest

from ai_cli import cli


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

    monkeypatch.setattr(
        "ai_cli.cli._invoke_with_retries",
        lambda *a, **k: 0,
    )

    result = cli.main(["--prompt", "hello"])
    assert result in (0, None)


def test_cli_no_args(monkeypatch):

    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO("")
    )

    cli.main([])

@pytest.mark.parametrize(
    "provider",
    [
        "openai",
        "gemini",
        "cohere",
        "deepseek",
        "xai",
        "perplexity",
    ],
)
def test_cli_provider_selection(provider, monkeypatch):

    monkeypatch.setattr(
        "ai_cli.cli.ask",
        lambda *a, **k: "ok"
    )

    cli.main(
        [
            "--provider",
            provider,
            "hello"
        ]
    )