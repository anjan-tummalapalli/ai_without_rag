from typing import Any


class ChatService:
    def __init__(self, provider: Any):
        self.provider = provider

    def ask(self, prompt: str) -> str:
        return self.provider.ask(prompt)