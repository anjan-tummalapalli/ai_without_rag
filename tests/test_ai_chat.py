"""
Tests for ai_cli.ai_chat module.

This test module is written to be production-grade:
- Clear, explicit docstrings for each test describing purpose,
    scenario and expectations.
- Strong typing and small helper utilities to reduce duplication.
- Integration tests are marked and timeboxed to avoid CI hangs.
- Defensive assertions include informative messages to aid debugging
    when a test fails.
"""

from __future__ import annotations

from typing import Optional

import pytest

from ai_cli.ai_chat import AVAILABLE_MODELS, PROVIDERS, ask

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VALID_PROVIDER = "openai"
VALID_PROMPT = "What is the capital of France?"
EXPECTED_RESPONSE_KEYWORD = "Paris"
INVALID_PROVIDER = "invalid_provider"
VALID_MODEL = "gpt-5.5"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def is_error_response(resp: str) -> bool:
        """Return True when the provider returned an error-style string.

        The ai_chat.ask() contract used in these tests returns textual errors
        prefixed with "[ERROR]". This helper centralises that knowledge for
        clarity.
        """
        return isinstance(resp, str) and resp.startswith("[ERROR]")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def valid_prompt() -> str:
        """Provide a canonical prompt used by many tests.

        This fixture isolates the canonical prompt so it can be updated in one
        place if needed.
        """
        return VALID_PROMPT


# -----------------------------------------------------------------------------
# ask() Tests
# -----------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_ask_valid_provider(valid_prompt: str) -> None:
        """Integration smoke test: valid provider should return a response.

        Scenario:
        - Call ask() with a known-good provider and a non-empty prompt.

        Expectations:
        - Non-empty string is returned.
        - The response contains an expected keyword (case-insensitive).
        - The response is not an error string.
        """
        response = ask(VALID_PROVIDER, valid_prompt)
        assert isinstance(response, str), "Expected a string response from ask()"
        assert response, "Expected non-empty response from provider"
        assert not is_error_response(response), (
                f"Provider returned error response: {response!r}"
        )
        assert EXPECTED_RESPONSE_KEYWORD.lower() in response.lower(), (
                f"Expected response to contain '{EXPECTED_RESPONSE_KEYWORD}' "
                f"but got: {response!r}"
        )


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_ask_with_model_override(valid_prompt: str) -> None:
        """Ensure explicit model override is respected and returns a valid answer.

        Scenario:
        - Call ask() with a valid provider, valid prompt and an explicit
            model override.

        Expectations:
        - A non-error, non-empty string is returned.
        """
        response = ask(VALID_PROVIDER, valid_prompt, model=VALID_MODEL)
        assert isinstance(response, str), (
                "Expected a string response when model override is used"
        )
        assert response, "Expected non-empty response when model override is used"
        assert not is_error_response(response), (
                f"Provider/model returned error response: {response!r}"
        )
        assert EXPECTED_RESPONSE_KEYWORD.lower() in response.lower(), (
                "Expected override model response to contain the expected keyword"
        )


def test_ask_invalid_provider() -> None:
        """Ensure that asking with an unknown provider returns an error string.

        Scenario:
        - ask() is called with an unknown provider name.

        Expectations:
        - The function returns a string and it starts with the configured
            error prefix.
        """
        response = ask(INVALID_PROVIDER, VALID_PROMPT)
        assert isinstance(response, str), (
                "When provider is invalid, ask() must still return a string"
        )
        assert is_error_response(response), (
                "Expected an error-style response for unknown provider"
        )


@pytest.mark.parametrize("prompt", ["", " ", "\n", "\t"])
def test_ask_invalid_prompt(prompt: str) -> None:
        """Invalid prompts should be handled gracefully and return an error.

        Scenario:
        - ask() invoked with whitespace or empty prompts.

        Expectations:
        - A string starting with the error prefix is returned rather than
            raising.
        """
        response = ask(VALID_PROVIDER, prompt)
        assert isinstance(response, str), "ask() must return a string for invalid prompts"
        assert is_error_response(response), (
                f"Expected error response for invalid prompt, got: {response!r}"
        )


# -----------------------------------------------------------------------------
# Provider Registry Tests
# -----------------------------------------------------------------------------
def test_available_models_structure() -> None:
        """Validate AVAILABLE_MODELS structure and that test provider exists.

        Expectations:
        - AVAILABLE_MODELS is a non-empty mapping.
        - The canonical provider used in tests is present and exposes at least
            one model.
        """
        assert isinstance(AVAILABLE_MODELS, dict), "AVAILABLE_MODELS must be a dict"
        assert AVAILABLE_MODELS, "AVAILABLE_MODELS must not be empty"
        assert VALID_PROVIDER in AVAILABLE_MODELS, (
                f"Expected {VALID_PROVIDER!r} to be registered in AVAILABLE_MODELS"
        )
        provider_models = AVAILABLE_MODELS[VALID_PROVIDER]
        assert isinstance(provider_models, (list, tuple)), (
                "Provider models entry must be a list or tuple"
        )
        assert provider_models, "Provider must expose at least one model"
        # Optional: if our canonical VALID_MODEL should exist, check if present.
        if VALID_MODEL:
                assert (
                        VALID_MODEL in provider_models
                        or any(VALID_MODEL in str(m) for m in provider_models)
                ), (
                        f"Expected VALID_MODEL {VALID_MODEL!r} to be discoverable for "
                        f"provider {VALID_PROVIDER}"
                )


def test_providers_registry_structure() -> None:
        """Ensure the PROVIDERS registry is sane.

        Expectations:
        - PROVIDERS is a non-empty dict and contains common providers.
        """
        assert isinstance(PROVIDERS, dict), "PROVIDERS must be a dict"
        assert PROVIDERS, "PROVIDERS registry must not be empty"
        required_providers = {"openai", "gemini"}
        assert required_providers.issubset(PROVIDERS.keys()), (
                f"Expected providers {required_providers} to be present in registry"
        )


# -----------------------------------------------------------------------------
# Negative / Edge Case Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("provider", [None, "", " ", "unknown"])
def test_invalid_provider_inputs(provider: Optional[str]) -> None:
        """Malformed provider inputs should be rejected with an error string.

        This test covers None and various empty/whitespace/unknown values that
        could be passed by callers.
        """
        response = ask(provider, VALID_PROMPT)
        assert isinstance(response, str), (
                "ask() must return a string even for malformed provider input"
        )
        assert is_error_response(response), f"Expected an error response for {provider!r}"


@pytest.mark.parametrize("model", ["", "invalid-model"])
def test_invalid_model_override(model: str) -> None:
        """Invalid model overrides should not raise but return a safe string.

        Expectations:
        - ask() returns a string. It may be an error message or a safe fallback.
        """
        response = ask(VALID_PROVIDER, VALID_PROMPT, model=model)
        assert isinstance(response, str), (
                "ask() must return a string when an invalid model override is provided"
        )


# -----------------------------------------------------------------------------
# Performance / Smoke Tests
# -----------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.timeout(30)
def test_provider_response_time(valid_prompt: str) -> None:
        """Basic latency smoke test to detect regressions in responsiveness.

        This is intentionally lightweight and does not enforce strict SLA beyond
        the pytest timeout decoration. It ensures that a request completes and
        yields a non-empty response under normal CI conditions.
        """
        response = ask(VALID_PROVIDER, valid_prompt)
        assert isinstance(response, str), "Expected a string response in smoke test"
        assert response, "Expected non-empty response in smoke test"
