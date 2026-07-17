from typing import Any
 
from ai_cli.providers.base import BaseProvider
 
 
class EchoProvider(BaseProvider):
    provider_name = "echo"
 
    def __init__(
            self,
            config: dict[str, Any] | None = None,
            **kwargs: Any
            ) -> None:
        super().__init__(provider_name="echo", config=config, **kwargs)
        self.config = config or {}
 
    def chat(
            self,
            prompt: str,
            **kwargs: Any
            ) -> str:
        return prompt
 
    def send(
            self,
            prompt: str,
            **kwargs: Any
            ) -> str:
        return self.chat(prompt, **kwargs)