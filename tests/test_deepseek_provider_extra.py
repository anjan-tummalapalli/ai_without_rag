from unittest.mock import MagicMock

import pytest
from ai_cli.providers.deepseek_provider import DeepSeekProvider


def test_deepseek_missing_key():
    with pytest.raises(RuntimeError):
        DeepSeekProvider(api_key=None).send("hello")


def test_deepseek_success():
    class FakeResp:
        def __init__(self):
            self.choices = [
                type("C", (), {"message": type("M", (), {"content": "ok"})()})()
            ]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = FakeResp()

    p = DeepSeekProvider(api_key="x")
    p.client = fake_client

    assert p.send("hello") == "ok"
