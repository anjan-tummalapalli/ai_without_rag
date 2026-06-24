from unittest.mock import MagicMock
import pytest

from ai_cli.providers.openai_provider import OpenAIProvider   # adjust import


@pytest.fixture
def p():
    provider = OpenAIProvider()   # adjust constructor args if needed

    provider.client.embeddings.create = MagicMock(
        return_value=type(
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
    )

    provider.client.chat.completions.create = MagicMock(
        return_value=type(
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
    )

    return provider