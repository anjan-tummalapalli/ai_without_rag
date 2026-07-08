import pytest

from ai_cli.core.prompt_corrector import correct_prompt


def test_prompt_corrector_basic():
    out = correct_prompt("  hello   world  ")
    assert isinstance(out, str)
    assert "hello" in out


@pytest.mark.parametrize("text", ["", "   ", None, "hello world"])
def test_prompt_corrector_inputs(text):
    result = correct_prompt(text)
    if text is None:
        assert result is None
    else:
        assert result is not None
