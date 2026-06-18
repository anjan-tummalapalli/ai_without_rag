from ai_cli.core.api import ask


class FakeProvider:
    def send(self, prompt):
        return "mock response"

def test_ask_basic():
    result = ask(
                 prompt="hello",
                 provider="openai",
                 model="gpt-4o-mini",
                 _provider=FakeProvider()
                )
    assert result == "mock response"