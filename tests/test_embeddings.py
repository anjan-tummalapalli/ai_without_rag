from unittest.mock import MagicMock, patch

import pytest

from ai_cli.providers.openai_provider import OpenAIProvider


@pytest.fixture
def p():
    mock_client = MagicMock()

    mock_client.embeddings.create.return_value = type(
        "R",
        (),
        {
            "data": [
                type(
                    "D",
                    (),
                    {"embedding": [0.1, 0.2]}
                )()
            ]
        }
    )()

    mock_client.chat.completions.create.return_value = type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {
                        "message": type(
                            "M",
                            (),
                            {"content": "test response"}
                        )()
                    }
                )()
            ]
        }
    )()

    with patch(
        "ai_cli.providers.openai_provider.OpenAI",
        return_value=mock_client,
    ):
        provider = OpenAIProvider(api_key="test-key")

    return provider


def test_send(p):
    result = p.send("hello")
    assert result == "test response"


def test_health_check(p):
    assert p.health_check() is True