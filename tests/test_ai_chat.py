from __future__ import annotations

import pytest

from ai_cli.ai_chat import (
    AVAILABLE_MODELS,
    PROVIDERS,
    ask,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

VALID_PROVIDER = "openai"

VALID_PROMPT = (
    "What is the capital of France?"
)

EXPECTED_RESPONSE_KEYWORD = "Paris"

INVALID_PROVIDER = "invalid_provider"

VALID_MODEL = "gpt-5.5"

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def valid_prompt() -> str:
    """
    Standard reusable test prompt.
    """
    return VALID_PROMPT


# -----------------------------------------------------------------------------
# ask() Tests
# -----------------------------------------------------------------------------


@pytest.mark.integration
def test_ask_valid_provider(
    valid_prompt: str,
) -> None:
    """
    Ensure valid providers return responses.
    """

    response = ask(
        VALID_PROVIDER,
        valid_prompt,
    )

    assert response
    assert isinstance(response, str)

    assert (
        EXPECTED_RESPONSE_KEYWORD
        in response
    )


@pytest.mark.integration
def test_ask_with_model_override(
    valid_prompt: str,
) -> None:
    """
    Ensure explicit model override works.
    """

    response = ask(
        VALID_PROVIDER,
        valid_prompt,
        model=VALID_MODEL,
    )

    assert response
    assert isinstance(response, str)

    assert (
        EXPECTED_RESPONSE_KEYWORD
        in response
    )


def test_ask_invalid_provider() -> None:
    """
    Ensure invalid providers fail gracefully.
    """

    response = ask(
        INVALID_PROVIDER,
        VALID_PROMPT,
    )

    assert isinstance(response, str)

    assert response.startswith(
        "[ERROR]"
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_ask_invalid_prompt(
    prompt: str,
) -> None:
    """
    Ensure invalid prompts fail safely.
    """

    response = ask(
        VALID_PROVIDER,
        prompt,
    )

    assert isinstance(response, str)

    assert response.startswith(
        "[ERROR]"
    )


# -----------------------------------------------------------------------------
# Provider Registry Tests
# -----------------------------------------------------------------------------


def test_available_models_structure() -> None:
    """
    Ensure AVAILABLE_MODELS structure is valid.
    """

    assert isinstance(
        AVAILABLE_MODELS,
        dict,
    )

    assert AVAILABLE_MODELS

    assert VALID_PROVIDER in AVAILABLE_MODELS

    assert isinstance(
        AVAILABLE_MODELS[
            VALID_PROVIDER
        ],
        (list, tuple),
    )

    assert (
        len(
            AVAILABLE_MODELS[
                VALID_PROVIDER
            ]
        )
        > 0
    )


def test_providers_registry_structure() -> None:
    """
    Ensure provider registry integrity.
    """

    assert isinstance(
        PROVIDERS,
        dict,
    )

    assert PROVIDERS

    required_providers = {
        "openai",
        "gemini",
    }

    assert required_providers.issubset(
        PROVIDERS.keys()
    )


# -----------------------------------------------------------------------------
# Negative / Edge Case Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    [
        None,
        "",
        " ",
        "unknown",
    ],
)
def test_invalid_provider_inputs(
    provider,
) -> None:
    """
    Ensure malformed providers are handled.
    """

    response = ask(
        provider,
        VALID_PROMPT,
    )

    assert isinstance(response, str)

    assert response.startswith(
        "[ERROR]"
    )


@pytest.mark.parametrize(
    "model",
    [
        "",
        "invalid-model",
    ],
)
def test_invalid_model_override(
    model: str,
) -> None:
    """
    Ensure invalid models fail safely.
    """

    response = ask(
        VALID_PROVIDER,
        VALID_PROMPT,
        model=model,
    )

    assert isinstance(response, str)


# -----------------------------------------------------------------------------
# Performance / Smoke Tests
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_provider_response_time(
    valid_prompt: str,
) -> None:
    """
    Basic provider latency smoke test.
    """

    response = ask(
        VALID_PROVIDER,
        valid_prompt,
    )

    assert response