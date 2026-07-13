import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider


def test_openai_provider_raises_on_api_error():
    class Boom(Exception):
        pass

    def fail(*args, **kwargs):
        raise Boom("fail")

    p = OpenAIProvider(api_key="x")
    p.client = type(
        "C",
        (),
        {
            "chat": type(
                "Chat",
                (),
                {
                    "completions": type(
                        "X",
                        (),
                        {"create": fail},
                    )()
                },
            )()
        },
    )()

    with pytest.raises(
        ProviderRequestError,
        match="OpenAI request failed",
    ):
        p.send("hello")

class Boom(Exception):
    pass

def fail(*args, **kwargs):
    raise Boom("fail")


def test_xai_missing_key():
    with pytest.raises((TypeError, ValueError, Exception)):
        XAIProvider(api_key=None).send("hello")


def test_zai_missing_key():
    with pytest.raises((TypeError, ValueError, Exception)):
        ZAIProvider(api_key=None).send("hello")
