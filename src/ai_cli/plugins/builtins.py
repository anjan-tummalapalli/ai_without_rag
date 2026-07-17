from __future__ import annotations
 
import logging
import os
from typing import Any
 
from ai_cli.core.exceptions import (
    ProviderConfigurationError,
    ProviderRequestError,
    ResponseValidationError,
)
from ai_cli.providers.base import AIProvider
from ai_cli.providers.registry import PROVIDERS, register_provider
 
logger = logging.getLogger(__name__)
 
 
class OpenAIProvider(AIProvider):
    """Concrete provider using OpenAI SDK."""
 
    def __init__(
            self,
            model: str | None = None
            ) -> None:
        super().__init__(
            provider_name="openai",
            model=model,
            provider_meta=PROVIDERS["openai"],
        )
        self.timeout = 60.0
 
    def send(
            self,
            prompt: str,
            **kwargs: Any
            ) -> str:
        try:
            import importlib
 
            OpenAI = importlib.import_module("openai").OpenAI
        except Exception as exc:
            raise ProviderConfigurationError("Install openai package") from exc
 
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY not set")
 
        client = OpenAI(api_key=api_key, timeout=self.timeout)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(f"OpenAI request failed: {exc}") from exc
 
        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ResponseValidationError("Invalid response structure") from exc
        if not content or not isinstance(content, str):
            raise ResponseValidationError("Empty response")
        return str(content).strip()
 
 
class OpenAICompatibleProvider(AIProvider):
    """Generic provider for OpenAI-compatible APIs."""
 
    api_base_url: str = ""
    api_key_env: str = ""
 
    def __init__(
            self, 
            provider_name: str,
            model: str | None = None
            ) -> None:
        super().__init__(
            provider_name=provider_name,
            model=model,
            provider_meta=PROVIDERS[provider_name],
        )
        self.provider_name = provider_name
        self.timeout = 60.0
 
    def _get_openai_client(self) -> Any:
        try:
            import importlib
 
            OpenAI = importlib.import_module("openai").OpenAI
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install OpenAI SDK: pip install openai"
            ) from exc
 
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderConfigurationError(f"{self.api_key_env} not set")
 
        return OpenAI(
            api_key=api_key, base_url=self.api_base_url, timeout=self.timeout
        )
 
    def send(
            self,
            prompt: str,
            **kwargs: Any
            ) -> str:
        client = self._get_openai_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                timeout=self.timeout,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(
                f"{self.provider_name} request failed: {exc}"
            ) from exc
 
        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise ResponseValidationError("Invalid response structure") from exc
        if not content or not isinstance(content, str):
            raise ResponseValidationError("Empty response")
        return str(content).strip()
 
 
class PerplexityProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.perplexity.ai"
    api_key_env = "PERPLEXITY_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="perplexity", model=model)
 
 
class DeepSeekProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.deepseek.com/v1"
    api_key_env = "DEEPSEEK_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="deepseek", model=model)
 
 
class GroqProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.groq.com/openai/v1"
    api_key_env = "GROQ_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="groq", model=model)
 
 
class OpenRouterProvider(OpenAICompatibleProvider):
    api_base_url = "https://openrouter.ai/api/v1"
    api_key_env = "OPENROUTER_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="openrouter", model=model)
 
 
class TogetherProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.together.xyz/v1"
    api_key_env = "TOGETHER_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="together", model=model)
 
 
class FireworksProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.fireworks.ai/inference/v1"
    api_key_env = "FIREWORKS_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="fireworks", model=model)
 
 
class XAIProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.x.ai/v1"
    api_key_env = "XAI_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="xai", model=model)
 
 
class GeminiProvider(OpenAICompatibleProvider):
    api_base_url = "https://api.gemini.google/v1"
    api_key_env = "GEMINI_API_KEY"
 
    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="gemini", model=model)
 
 
class CohereProvider(AIProvider):
    api_key_env = "COHERE_API_KEY"
 
    def __init__(
            self,
            model: str | None = None
            ) -> None:
        super().__init__(
            provider_name="cohere",
            model=model,
            provider_meta=PROVIDERS["cohere"],
        )
        self.timeout = 60.0
 
    def send(
            self,
            prompt: str,
            **kwargs: Any
            ) -> str:
        try:
            import importlib
 
            cohere = importlib.import_module("cohere")
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install cohere package: pip install cohere"
            ) from exc
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise ProviderConfigurationError(f"{self.api_key_env} not set")
 
        client = cohere.Client(api_key)
        try:
            response = client.generate(
                model=self.model,
                prompt=prompt,
                max_tokens=2048,
            )
        except Exception as exc:
            raise ProviderRequestError(f"Cohere request failed: {exc}") from exc
 
        try:
            content = response.generations[0].text
        except Exception as exc:
            raise ResponseValidationError("Invalid response structure") from exc
 
        if not content or not isinstance(content, str):
            raise ResponseValidationError("Empty response")
 
        return str(content).strip()
 
 
# Register them
register_provider("openai", OpenAIProvider, PROVIDERS["openai"])
register_provider("perplexity", PerplexityProvider, PROVIDERS["perplexity"])
register_provider("cohere", CohereProvider, PROVIDERS["cohere"])
register_provider("deepseek", DeepSeekProvider, PROVIDERS["deepseek"])
register_provider("groq", GroqProvider, PROVIDERS["groq"])
register_provider("openrouter", OpenRouterProvider, PROVIDERS["openrouter"])
register_provider("together", TogetherProvider, PROVIDERS["together"])
register_provider("fireworks", FireworksProvider, PROVIDERS["fireworks"])
register_provider("xai", XAIProvider, PROVIDERS["xai"])
register_provider("gemini", GeminiProvider, PROVIDERS["gemini"])
