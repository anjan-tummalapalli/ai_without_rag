from ai_cli.core.prompt_corrector import correct_prompt


def test_prompt_corrector_basic():
    out = correct_prompt("  hello   world  ")
    assert isinstance(out, str)
    assert "hello" in out