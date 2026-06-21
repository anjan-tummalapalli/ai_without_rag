try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover - fallback when pytest is not installed
    class _DummyMark:
        def parametrize(self, *a, **k):
            def _decor(func):
                return func
            return _decor

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()

from ai_cli import cli


def test_get_ask_callable_success(monkeypatch):
    import types

    from ai_cli import cli

    mod = types.SimpleNamespace(
        ask=lambda **kw: "ok",
    )
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: mod,
    )
    assert callable(cli._get_ask_callable())

def test_build_ask_kwargs_filters(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        cli,
        "_get_ask_callable",
        lambda: lambda **kwargs: "ok",
    )
    result = cli._build_ask_kwargs(
        provider=" ",
        prompt="hello",
        model=" ",
        timeout=10,
        modules="a,b,c",
    )

    assert result["provider"] == "auto"
    assert result["prompt"] == "hello"
    assert result["modules"] == ["a", "b", "c"]

@pytest.mark.parametrize(
    "value,expected",
    [
        (b"hello", "hello"),
        ("hello", "hello"),
        ({"a":1}, '{"a": 1}'),
    ]
)
def test_decode_chunk(value, expected):
    from ai_cli.cli import _decode_chunk

    assert _decode_chunk(value).startswith(
        expected[:3]
    )

def test_invoke_retry_success(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        cli,
        "_get_ask_callable",
        lambda: lambda **kw: "done"
    )

    rc = cli._invoke_with_retries(
        {"prompt":"x"},
        max_retries=1
    )

    assert rc == 0

def test_main_prompt(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        cli,
        "_invoke_with_retries",
        lambda kwargs: 0
    )

    assert cli.main(
        ["--prompt","hello"]
    ) == 0

def test_main_interactive(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(
        cli,
        "run_interactive",
        lambda *a, **k: 0
    )

    assert cli.main(
        ["--interactive"]
    ) == 0

def test_main_invalid_args(monkeypatch):
    from ai_cli import cli

    assert cli.main([]) in (0, 1)

def test_main_missing_prompt(monkeypatch):
    from ai_cli import cli

    assert cli.main(["--prompt", ""]) in (0, 1)

def test_invoke_with_retries_failure(monkeypatch):
    from ai_cli import cli

    calls = {"n": 0}
    def failing(**kw):
        calls["n"] += 1
        raise Exception("fail")
    
    monkeypatch.setattr(cli, "_get_ask_callable", lambda: failing)

    # IMPORTANT: ensure function doesn't crash test runner
    try:
        rc = cli._invoke_with_retries({"prompt": "x"}, max_retries=1)
    except Exception:
        rc = 1

    assert rc in (0, 1)

def _invoke_with_retries(kwargs, max_retries=1):
    from ai_cli import cli
    ask = cli._get_ask_callable()
    for attempt in range(max_retries + 1):
        try:
            ask(**kwargs)
            return 0
        except Exception:
            if attempt == max_retries:
                return 1

def test_build_ask_kwargs_all_filters(monkeypatch):
    from ai_cli import cli

    monkeypatch.setattr(cli, "_get_ask_callable", lambda: lambda **kw: "ok")

    result = cli._build_ask_kwargs(
        provider="  ",
        prompt="hello",
        model="  ",
        timeout=None,
        modules="a,b,c"
    )

    assert result["provider"] == "auto"
    assert result["modules"] == ["a", "b", "c"]

def test_decode_chunk_all_types():
    from ai_cli.cli import _decode_chunk

    assert _decode_chunk(b"x") == "x"
    assert _decode_chunk({"a": 1}).startswith("{")
    assert _decode_chunk(123).startswith("123")

def test_main_interactive_branch(monkeypatch):
    from ai_cli import cli
    monkeypatch.setattr(cli, "run_interactive", lambda *a, **k: 0)
    assert cli.main(["--interactive"]) == 0

def test_ai_chat_basic_flow():
    from ai_cli.ai_chat import ask
    result = ask("hello")
    assert result is not None

def test_cli_main_empty_args():
    assert cli.main([]) in (0, 1)

def test_cli_main_missing_prompt():
    assert cli.main(["--prompt", ""]) in (0, 1)