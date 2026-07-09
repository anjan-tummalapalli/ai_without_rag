from unittest.mock import patch

import pytest
from ai_cli.core.exceptions import ResponseValidationError
from ai_cli.providers.registry import (
    CHAT_PROVIDERS,
    PROVIDER_MAP,
    PROVIDERS,
    build_provider,
    get_chat_provider,
    list_providers,
    register_chat_provider,
    register_provider,
)
from ai_cli.utils.validation import (
    HallucinationDetector,
    ResponseValidator,
)


def test_hallucination_empty_response():
    result = HallucinationDetector().evaluate("")

    assert result.score > 0
    assert result.passed is True


def test_hallucination_suspicious_phrase():
    result = HallucinationDetector().evaluate("This is 100% accurate")

    assert result.score > 0
    assert "suspicious phrase" in result.reasons[0]


def test_hallucination_todo():
    result = HallucinationDetector().evaluate("TODO implement this")

    assert result.passed is False


def test_response_validator_empty():
    validator = ResponseValidator()

    try:
        validator.validate("")
        raise AssertionError("Expected ResponseValidationError")
    except ResponseValidationError:
        assert True


class DummyProvider:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def setup_function():
    PROVIDER_MAP.clear()
    CHAT_PROVIDERS.clear()


def test_register_provider_direct_registration():
    register_provider("dummy", DummyProvider)

    assert PROVIDER_MAP["dummy"] is DummyProvider


def test_register_provider_decorator_registration():
    @register_provider("decorator")
    class MyProvider:
        pass

    assert PROVIDER_MAP["decorator"] is MyProvider


def test_register_chat_provider_registers_both_maps():
    register_chat_provider("chat", DummyProvider)

    assert CHAT_PROVIDERS["chat"] is DummyProvider
    assert PROVIDER_MAP["chat"] is DummyProvider


@patch("ai_cli.providers.registry.ensure_initialized")
def test_get_chat_provider_success(mock_init):
    register_chat_provider("chat", DummyProvider)

    obj = get_chat_provider("chat", foo="bar")

    mock_init.assert_called_once()
    assert isinstance(obj, DummyProvider)
    assert obj.kwargs == {"foo": "bar"}


@patch("ai_cli.providers.registry.ensure_initialized")
def test_get_chat_provider_unknown(mock_init):
    with pytest.raises(ValueError, match="Unknown chat provider"):
        get_chat_provider("missing")


@patch("ai_cli.providers.registry.ensure_initialized")
def test_list_providers_sorted(mock_init):
    register_provider("z", DummyProvider)
    register_provider("a", DummyProvider)

    assert list_providers() == ["a", "z"]


@patch("ai_cli.providers.registry.ensure_initialized")
def test_build_provider_success(mock_init):
    register_provider("dummy", DummyProvider)

    obj = build_provider("dummy", answer=42)

    assert isinstance(obj, DummyProvider)
    assert obj.kwargs["answer"] == 42


@patch("ai_cli.providers.registry.ensure_initialized")
def test_build_provider_unknown(mock_init):
    with pytest.raises(ValueError, match="Unknown provider"):
        build_provider("missing")


def test_provider_registry_getitem_returns_registered_class():
    register_provider("dummy", DummyProvider)

    assert PROVIDERS["dummy"] is DummyProvider
    assert PROVIDERS["does-not-exist"] is None
