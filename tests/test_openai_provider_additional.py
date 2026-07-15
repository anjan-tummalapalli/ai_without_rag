from unittest.mock import MagicMock

import pytest
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.openai_provider import OpenAIProvider


def test_openai_provider_raises_on_api_error(monkeypatch):
    p = OpenAIProvider(api_key="x")

    class Boom(Exception):
        pass

    def fail(*args, **kwargs):
        raise Boom("fail")

    client = MagicMock()
    client.chat.completions.create.side_effect = fail
    monkeypatch.setattr(p, "client", client, raising=False)

    with pytest.raises(ProviderRequestError, match="OpenAI request failed"):
        p.send("hello")
