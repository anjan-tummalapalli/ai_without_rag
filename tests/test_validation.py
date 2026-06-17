from ai_cli.core.exceptions import ResponseValidationError
from ai_cli.utils.validation import (
    HallucinationDetector,
    ResponseValidator,
)


def test_hallucination_empty_response():
    result = HallucinationDetector().evaluate("")

    assert result.score > 0
    assert result.passed is True


def test_hallucination_suspicious_phrase():
    result = HallucinationDetector().evaluate(
        "This is 100% accurate"
    )

    assert result.score > 0
    assert "suspicious phrase" in result.reasons[0]


def test_hallucination_todo():
    result = HallucinationDetector().evaluate(
        "TODO implement this"
    )

    assert result.passed is False


def test_response_validator_empty():
    validator = ResponseValidator()

    try:
        validator.validate("")
        raise AssertionError("Expected ResponseValidationError")
    except ResponseValidationError:
        assert True

