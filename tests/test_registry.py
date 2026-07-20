import pytest
from ai_cli.core.exceptions import ResponseValidationError
from ai_cli.providers.registry import PROVIDER_MAP, build_provider
from ai_cli.utils.validation import (
    HallucinationDetector,
    HallucinationResult,
    ResponseValidator,
)


def test_build_provider_invalid():
    with pytest.raises(ValueError):
        build_provider("unknown_provider")


def test_build_provider_normal(monkeypatch):
    class DummyProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(PROVIDER_MAP, "dummy", DummyProvider)
    provider = build_provider("dummy", foo="bar")
    assert isinstance(provider, DummyProvider)
    assert provider.kwargs["foo"] == "bar"


def test_hallucination_empty_response():
    detector = HallucinationDetector()

    result = detector.evaluate("")

    assert isinstance(result, HallucinationResult)
    assert result.score == 0.4
    assert result.passed is True
    assert "response too short" in result.reasons


def test_hallucination_short_response():
    detector = HallucinationDetector()

    result = detector.evaluate("abc")

    assert result.score == 0.4
    assert result.passed is True


def test_hallucination_single_pattern():
    detector = HallucinationDetector()

    result = detector.evaluate("This is guaranteed to succeed.")

    assert result.score == 0.2
    assert result.passed is True
    assert any("guaranteed" in r for r in result.reasons)


def test_hallucination_all_patterns():
    detector = HallucinationDetector()

    text = "100% accurate guaranteed always works never fails trust me"

    result = detector.evaluate(text)

    assert result.score == 1.0
    assert result.passed is False
    assert len(result.reasons) == 5


def test_hallucination_placeholder_only():
    detector = HallucinationDetector()

    result = detector.evaluate("TODO")

    assert result.score == 1.0
    assert result.passed is False
    assert "placeholder content detected" in result.reasons


def test_hallucination_clean_response():
    detector = HallucinationDetector()

    result = detector.evaluate(
        "This explains the feature in a clear and detailed way."
    )

    assert result.score == 0
    assert result.passed is True
    assert result.reasons == []


def test_response_validator_empty():
    validator = ResponseValidator()

    with pytest.raises(ResponseValidationError, match="empty response"):
        validator.validate("")


def test_response_validator_short():
    validator = ResponseValidator()

    with pytest.raises(ResponseValidationError, match="response too short"):
        validator.validate("abc")


def test_response_validator_valid():
    validator = ResponseValidator()

    validator.validate("This is definitely long enough.")
