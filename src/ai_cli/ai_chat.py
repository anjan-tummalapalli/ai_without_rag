from __future__ import annotations

"""
===============================================================================
Enterprise AI Gateway
===============================================================================

A production-oriented multi-provider AI gateway framework supporting multiple
LLM vendors through a unified interface.

-------------------------------------------------------------------------------
FEATURES
-------------------------------------------------------------------------------

- Unified provider abstraction
- Multi-provider AI routing
- Prompt validation and sanitization
- Retry engine with exponential backoff
- Structured logging
- Runtime metrics collection
- Hallucination heuristic detection
- Response validation
- OpenAI-compatible provider support
- CLI support
- Token usage tracking
- Extensible provider architecture

-------------------------------------------------------------------------------
SUPPORTED PROVIDERS
-------------------------------------------------------------------------------

Native SDK Providers
--------------------

1. OpenAI
2. Google Gemini

OpenAI-Compatible Providers
---------------------------

3. Perplexity
4. DeepSeek
5. Groq
6. xAI (Grok)
7. Together AI
8. OpenRouter
9. Fireworks AI

Planned Providers
-----------------

10. Anthropic
11. Cohere
12. Mistral
13. Ollama
14. Hugging Face

-------------------------------------------------------------------------------
INSTALLATION
-------------------------------------------------------------------------------

Core Dependencies
-----------------

pip install openai google-generativeai

Optional Provider SDKs
----------------------

OpenAI:
    pip install openai

Gemini:
    pip install google-generativeai

Recommended Full Installation
-----------------------------

pip install openai google-generativeai requests

-------------------------------------------------------------------------------
ENVIRONMENT VARIABLES
-------------------------------------------------------------------------------

OpenAI
-------
OPENAI_API_KEY=your_key

Gemini
-------
GEMINI_API_KEY=your_key

Perplexity
-----------
PERPLEXITY_API_KEY=your_key

DeepSeek
---------
DEEPSEEK_API_KEY=your_key

Groq
----
GROQ_API_KEY=your_key

xAI
---
XAI_API_KEY=your_key

Together AI
------------
TOGETHER_API_KEY=your_key

OpenRouter
-----------
OPENROUTER_API_KEY=your_key

Fireworks AI
-------------
FIREWORKS_API_KEY=your_key

-------------------------------------------------------------------------------
CLI USAGE
-------------------------------------------------------------------------------

Basic Usage
-----------

python ai_gateway.py --provider openai --prompt "Hello"

Specify Model
-------------

python ai_gateway.py \
    --provider gemini \
    --model gemini-2.5-pro \
    --prompt "Explain quantum computing"

Read Prompt from STDIN
----------------------

echo "Explain AI agents" | python ai_gateway.py --provider groq

-------------------------------------------------------------------------------
PYTHON API USAGE
-------------------------------------------------------------------------------

Simple Usage
------------
from ai_gateway import ask

response = ask(
    provider="openai",
    prompt="Explain transformers"
)
print(response)

-------------------------------------------------------------------------------
DIRECT PROVIDER USAGE
-------------------------------------------------------------------------------

OpenAI Example
--------------

provider = OpenAIProvider(
    model="gpt-5.5"
)

response = provider.send(
    "Write Python code"
)

Gemini Example
--------------

provider = GeminiProvider(
    model="gemini-2.5-pro"
)

response = provider.send(
    "Explain neural networks"
)

Groq Example
------------

provider = GroqProvider(
    model="llama-3.3-70b"
)

response = provider.send(
    "Generate SQL query"
)

-------------------------------------------------------------------------------
LATEST RECOMMENDED MODELS
-------------------------------------------------------------------------------

OpenAI
-------
- gpt-5.5
- gpt-4.1
- gpt-4o

Gemini
-------
- gemini-2.5-pro
- gemini-2.5-flash

Perplexity
-----------
- sonar
- sonar-pro

DeepSeek
---------
- deepseek-chat
- deepseek-coder
- deepseek-reasoner

Groq
----
- llama-3.3-70b
- mixtral-8x7b

xAI
---
- grok-3
- grok-3-mini

Together AI
------------
- meta-llama/Llama-3-70b-chat-hf
- mistralai/Mixtral-8x7B-Instruct-v0.1

OpenRouter
-----------
- openai/gpt-4o
- anthropic/claude-3.5-sonnet
- google/gemini-2.5-pro

Fireworks AI
-------------
- accounts/fireworks/models/llama-v3p1-70b-instruct

-------------------------------------------------------------------------------
ARCHITECTURE
-------------------------------------------------------------------------------

AIProvider
    Abstract provider base class.

OpenAICompatibleProvider
    Generic provider implementation for OpenAI-compatible APIs.

GeminiProvider
    Native Gemini SDK implementation.

RetryEngine
    Handles retries with exponential backoff.

HallucinationDetector
    Performs lightweight heuristic hallucination analysis.

ResponseValidator
    Ensures valid provider responses.

-------------------------------------------------------------------------------
PROVIDER IMPLEMENTATION GUIDE
-------------------------------------------------------------------------------

To add a new provider:

1. Add metadata to PROVIDERS
2. Create provider class
3. Register provider in PROVIDER_MAP

Example:

class MyProvider(OpenAICompatibleProvider):

    api_base_url = "https://api.example.com/v1"
    api_key_env = "MY_API_KEY"

    def __init__(self, model=None):
        super().__init__(
            provider_name="myprovider",
            model=model,
        )

-------------------------------------------------------------------------------
PRODUCTION DEPLOYMENT RECOMMENDATIONS
-------------------------------------------------------------------------------
Recommended Enhancements
------------------------
- Redis caching
- Async request support
- Circuit breaker
- Streaming token support
- Distributed tracing
- Prometheus metrics
- Rate limiting
- API gateway integration
- Load balancing
- Secret management
- Kubernetes deployment
- Observability dashboard

-------------------------------------------------------------------------------
GITHUB ACTIONS CI/CD
-------------------------------------------------------------------------------
Example Workflow
----------------

name: AI Gateway CI

on:
  push:
    branches:
      - main
jobs:

  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: |
          pip install -r requirements.txt

      - run: |
          python ai_gateway.py \
            --provider openai \
            --prompt "hello"

-------------------------------------------------------------------------------
SECURITY NOTES
-------------------------------------------------------------------------------
- Never hardcode API keys
- Use secret managers in production
- Rotate credentials regularly
- Apply rate limiting
- Validate prompts before execution
- Monitor provider usage and cost
-------------------------------------------------------------------------------
LICENSE
-------------------------------------------------------------------------------
MIT License
===============================================================================
"""

import argparse
import logging
import os, random, re, sys, time, uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Final, TypeVar

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)

logger = logging.getLogger("ai_gateway")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TIMEOUT_SECONDS: Final[int] = 60
DEFAULT_MAX_PROMPT_LENGTH: Final[int] = 10_000
MIN_RESPONSE_LENGTH: Final[int] = 5

T = TypeVar("T")

# =============================================================================
# Exceptions
# =============================================================================


class AIProviderError(Exception):
    """
    Base exception for all AI provider errors.
    """


class PromptValidationError(AIProviderError):
    """
    Raised when prompt validation fails.
    """


class ProviderConfigurationError(AIProviderError):
    """
    Raised when provider configuration is invalid.
    """


class ProviderRequestError(AIProviderError):
    """
    Raised when provider request execution fails.
    """


class ResponseValidationError(AIProviderError):
    """
    Raised when AI response validation fails.
    """


# =============================================================================
# Provider Metadata
# =============================================================================


@dataclass(frozen=True)
class ProviderMetadata:
    """
    Immutable provider configuration.

    Attributes:
        name:
            Human-readable provider name.

        default_model:
            Default model used by provider.

        supported_models:
            List of supported model identifiers.

        supports_streaming:
            Whether provider supports token streaming.

        supports_tools:
            Whether provider supports function/tool calling.

        supports_vision:
            Whether provider supports image inputs.

        max_context:
            Maximum context window size.

        cost_per_1k_tokens:
            Approximate pricing per 1K tokens.

        avg_latency_ms:
            Average provider latency in milliseconds.
    """

    name: str
    default_model: str
    supported_models: list[str]
    supports_streaming: bool
    supports_tools: bool
    supports_vision: bool
    max_context: int
    cost_per_1k_tokens: float
    avg_latency_ms: int


# =============================================================================
# Provider Registry
# =============================================================================

PROVIDERS: dict[str, ProviderMetadata] = {
    "openai": ProviderMetadata(
        name="OpenAI",
        default_model="gpt-5.5",
        supported_models=[
            "gpt-5.5",
            "gpt-4.1",
            "gpt-4o",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=1_000_000,
        cost_per_1k_tokens=0.01,
        avg_latency_ms=800,
    ),
    "mistral": ProviderMetadata(
        name="Mistral AI",
        default_model="mistral-large",
        supported_models=[
            "mistral-large",
            "codestral",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.007,
        avg_latency_ms=650,
    ),
    "perplexity": ProviderMetadata(
        name="Perplexity AI",
        default_model="sonar-pro",
        supported_models=[
            "sonar",
            "sonar-pro",
        ],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.009,
        avg_latency_ms=850,
    ),
    "cohere": ProviderMetadata(
        name="Cohere",
        default_model="command-r-plus",
        supported_models=[
            "command-r",
            "command-r-plus",
        ],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.006,
        avg_latency_ms=600,
    ),
    "gemini": ProviderMetadata(
        name="Google Gemini",
        default_model="gemini-2.5-pro",
        supported_models=[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=1_000_000,
        cost_per_1k_tokens=0.008,
        avg_latency_ms=700,
    ),
    "xai": ProviderMetadata(
        name="xAI Grok",
        default_model="grok-3",
        supported_models=[
            "grok-3",
            "grok-3-mini",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        cost_per_1k_tokens=0.011,
        avg_latency_ms=750,
    ),
    "deepseek": ProviderMetadata(
            name="DeepSeek",
            default_model="deepseek-chat",
            supported_models=[
                "deepseek-chat",
                "deepseek-coder",
                "deepseek-reasoner",
            ],
            supports_streaming=True,
            supports_tools=True,
            supports_vision=False,
            max_context=128_000,
            cost_per_1k_tokens=0.003,
            avg_latency_ms=500,
    ),
    "groq": ProviderMetadata(
           name="Groq",
           default_model="llama-3.3-70b",
           supported_models=[
                "llama-3.3-70b",
                "mixtral-8x7b",
           ],
           supports_streaming=True,
           supports_tools=True,
           supports_vision=False,
           max_context=128_000,
           cost_per_1k_tokens=0.002,
           avg_latency_ms=300,
    ),
    "openrouter": ProviderMetadata(
           name="OpenRouter",
           default_model="openai/gpt-4o",
           supported_models=[
                "openai/gpt-4o",
                "anthropic/claude-3.5-sonnet",
                "google/gemini-2.5-pro",
           ],
           supports_streaming=True,
           supports_tools=True,
           supports_vision=True,
           max_context=200_000,
           cost_per_1k_tokens=0.005,
           avg_latency_ms=700,
    ),
    "together": ProviderMetadata(
           name="Together AI",
           default_model="meta-llama/Llama-3-70b-chat-hf",
           supported_models=[
                 "meta-llama/Llama-3-70b-chat-hf",
                 "mistralai/Mixtral-8x7B-Instruct-v0.1",
           ],
           supports_streaming=True,
           supports_tools=False,
           supports_vision=False,
           max_context=128_000,
           cost_per_1k_tokens=0.004,
           avg_latency_ms=650,
    ),
    "fireworks": ProviderMetadata(
           name="Fireworks AI",
           default_model="accounts/fireworks/models/llama-v3p1-70b-instruct",
           supported_models=[
               "accounts/fireworks/models/llama-v3p1-70b-instruct",
            ],
           supports_streaming=True,
           supports_tools=False,
           supports_vision=False,
           max_context=128_000,
           cost_per_1k_tokens=0.004,
           avg_latency_ms=550,
    ),
    "gemini": ProviderMetadata(
           name="Google Gemini",
           default_model="gemini-2.5-pro",
           supported_models=[
               "gemini-2.5-pro",
               "gemini-2.5-flash",
           ],
           supports_streaming=True,
           supports_tools=True,
           supports_vision=True,
           max_context=1_000_000,
           cost_per_1k_tokens=0.008,
           avg_latency_ms=700,
    )
}

# =============================================================================
# Metrics
# =============================================================================
@dataclass
class ModelQualityMetrics:
    """
    Runtime quality and operational metrics for a provider.

    Attributes:
        provider:
            Provider identifier.

        model:
            Active model name.

        requests:
            Total requests executed.

        failures:
            Total failed requests.

        total_latency_seconds:
            Cumulative latency across requests.

        hallucination_failures:
            Total hallucination detection failures.

        total_prompt_tokens:
            Total prompt tokens processed.

        total_completion_tokens:
            Total completion tokens generated.
    """

    provider: str
    model: str
    requests: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0
    hallucination_failures: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0

    @property
    def success_rate(self) -> float:
        """
        Calculate provider success rate.
        """

        if self.requests == 0:
            return 0.0

        return (
            self.requests - self.failures
        ) / self.requests

    @property
    def avg_latency(self) -> float:
        """
        Calculate average request latency.
        """

        if self.requests == 0:
            return 0.0

        return (
            self.total_latency_seconds
            / self.requests
        )

    @property
    def hallucination_rate(self) -> float:
        """
        Calculate hallucination detection rate.
        """

        if self.requests == 0:
            return 0.0

        return (
            self.hallucination_failures
            / self.requests
        )


# =============================================================================
# Hallucination Detection
# =============================================================================


@dataclass
class HallucinationResult:
    """
    Result returned from hallucination analysis.

    Attributes:
        score:
            Heuristic hallucination score.

        passed:
            Whether hallucination check passed.

        reasons:
            Detection reasons.
    """

    score: float
    passed: bool
    reasons: list[str] = field(default_factory=list)


class HallucinationDetector:
    """
    Lightweight heuristic-based hallucination detector.

    This detector is NOT a factual verifier.
    It performs pattern-based risk estimation.
    """

    SUSPICIOUS_PATTERNS: Final[list[str]] = [
        r"100% accurate",
        r"guaranteed",
        r"always works",
        r"never fails",
        r"trust me",
    ]

    def evaluate(self, response: str) -> HallucinationResult:
        """
        Evaluate response hallucination risk.
        """

        score = 0.0
        reasons: list[str] = []

        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            score += 0.4
            reasons.append("response too short")

        for pattern in self.SUSPICIOUS_PATTERNS:
            if re.search(
                pattern,
                response,
                re.IGNORECASE,
            ):
                score += 0.2
                reasons.append(f"suspicious phrase: {pattern}")

        if "TODO" in response:
            score += 0.3
            reasons.append("placeholder content detected")

        score = min(score, 1.0)

        return HallucinationResult(
            score=score,
            passed=score < 0.5,
            reasons=reasons,
        )


# =============================================================================
# Validation
# =============================================================================


class ResponseValidator:
    """
    Validates AI provider responses.
    """

    def validate(self, response: str) -> None:
        """
        Validate provider response.
        """

        if not response:
            raise ResponseValidationError(
                "empty response"
            )

        if len(response.strip()) < MIN_RESPONSE_LENGTH:
            raise ResponseValidationError(
                "response too short"
            )


# =============================================================================
# Retry Engine
# =============================================================================
class RetryEngine:
    """
    Retry executor with exponential backoff.

    Args:
        max_attempts:
            Maximum retry attempts.

        base_delay:
            Base retry delay in seconds.
    """

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def execute(self, func: Callable[[], T]) -> T:
        """
        Execute retryable operation.
        """
        last_error: Exception | None = None
        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                sleep_time = (self.base_delay * (2 ** (attempt - 1)))
                jitter = random.uniform(0, 0.5)

                logger.warning(
                    "retry_attempt=%s "
                    "sleep=%s "
                    "error=%s",
                    attempt,
                    sleep_time,
                    exc,
                )

                time.sleep(sleep_time + jitter)

        raise last_error  # type: ignore[misc]


# =============================================================================
# AI Provider Base Class
# =============================================================================
class AIProvider(ABC):
    """
    Abstract base class for AI providers.

    This class standardizes:

    - prompt validation
    - retry handling
    - metrics
    - response validation
    - hallucination analysis
    - structured logging

    Subclasses must implement `_send_impl()`.
    """

    def __init__(
        self,
        provider_name: str,
        model: str | None = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.provider_name = provider_name
        self.timeout = timeout
        provider_meta = PROVIDERS[provider_name]
        self.model = (model or provider_meta.default_model)
        self.trace_id = str(uuid.uuid4())
        self.retry_engine = RetryEngine()
        self.response_validator = (
            ResponseValidator()
        )
        self.hallucination_detector = (
            HallucinationDetector()
        )
        self.metrics = ModelQualityMetrics(
            provider=provider_name,
            model=self.model,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        """
        Return provider metadata.
        """
        return PROVIDERS[self.provider_name]

    def validate_prompt(self, prompt: str) -> str:
        """
        Validate and sanitize user prompt.
        """
        if not isinstance(prompt, str):
            raise PromptValidationError("prompt must be string")

        prompt = prompt.strip()
        if not prompt:
            raise PromptValidationError("prompt is empty")

        if len(prompt) > DEFAULT_MAX_PROMPT_LENGTH:
            raise PromptValidationError("prompt exceeds maximum length")

        if "\x00" in prompt:
            raise PromptValidationError("prompt contains NUL byte")

        sanitized = "".join(
            ch
            for ch in prompt
            if ch in ("\n", "\t")
            or ord(ch) >= 32
        )
        return sanitized

    @abstractmethod
    def _send_impl(self, prompt: str) -> str:
        """
        Provider-specific implementation.
        """

    def send(self, prompt: str) -> str:
        """
        Execute provider request.
        """

        validated_prompt = (
            self.validate_prompt(prompt)
        )

        self.metrics.requests += 1

        start_time = time.monotonic()

        logger.info(
            "provider_request "
            "provider=%s "
            "model=%s "
            "trace_id=%s",
            self.provider_name,
            self.model,
            self.trace_id,
        )

        try:
            response = self.retry_engine.execute(
                lambda: self._send_impl(validated_prompt)
            )

            duration = (
                time.monotonic() - start_time
            )

            self.metrics.total_latency_seconds += duration
            self.response_validator.validate(response)

            hallucination = (
                self.hallucination_detector.evaluate(
                    validated_prompt,
                    response,
                )
            )

            if not hallucination.passed:
                self.metrics.hallucination_failures += 1

                logger.warning(
                    "hallucination_detected "
                    "provider=%s "
                    "score=%s "
                    "reasons=%s",
                    self.provider_name,
                    hallucination.score,
                    hallucination.reasons,
                )
            return response.strip()

        except Exception as exc:
            self.metrics.failures += 1
            logger.exception(
                "provider_error "
                "provider=%s "
                "trace_id=%s "
                "error=%s",
                self.provider_name,
                self.trace_id,
                exc,
            )

            raise ProviderRequestError(
                f"{self.provider_name} request failed: {exc}"
            ) from exc


# =============================================================================
# Providers
# =============================================================================
class EchoProvider(AIProvider):
    """
    Local echo provider used for testing.
    """

    def __init__(self, model: str | None = None) -> None:
        super().__init__(
            provider_name="openai",
            model=model,
        )

    def _send_impl(self, prompt: str) -> str:
        return f"(echo) {prompt}"


class OpenAIProvider(AIProvider):
    """
    OpenAI provider integration.
    """

    def __init__(self, model: str | None = None) -> None:
        super().__init__(provider_name="openai", model=model)

    def _send_impl(self, prompt: str) -> str:
        try:
            from openai import OpenAI

        except Exception as exc:
            raise ProviderConfigurationError(
                "Install openai package"
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ProviderConfigurationError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        response = (
            client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                timeout=self.timeout,
                max_tokens=2048,
            )
        )
        usage = getattr(response, "usage", None)
        if usage:
            self.metrics.total_prompt_tokens += (
                getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
            )

            self.metrics.total_completion_tokens += (
                getattr(
                    usage,
                    "completion_tokens",
                    0,
                )
            )

        return (
            response
            .choices[0]
            .message
            .content
        )

# =============================================================================
# Perplexity Provider
# =============================================================================

class PerplexityProvider(OpenAICompatibleProvider):

    api_base_url = (
        "https://api.perplexity.ai"
    )

    api_key_env = (
        "PERPLEXITY_API_KEY"
    )

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        super().__init__(
            provider_name="perplexity",
            model=model,
        )

# =============================================================================
# DeepSeek Provider
# =============================================================================

class DeepSeekProvider(OpenAICompatibleProvider):

    api_base_url = (
        "https://api.deepseek.com/v1"
    )

    api_key_env = (
        "DEEPSEEK_API_KEY"
    )

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        super().__init__(
            provider_name="deepseek",
            model=model,
        )



# =============================================================================
# Groq Provider
# =============================================================================

class GroqProvider(OpenAICompatibleProvider):
    """
    Groq provider integration.
    """

    api_base_url = (
        "https://api.groq.com/openai/v1"
    )

    api_key_env = (
        "GROQ_API_KEY"
    )
    def __init__(self, model: str | None = None) -> None:
        super().__init__(
            provider_name="groq",
            model=model,
        )

# =============================================================================
# OpenRouter Provider
# =============================================================================

class OpenRouterProvider(OpenAICompatibleProvider):
    """
    OpenRouter provider integration.
    """

    api_base_url = (
        "https://openrouter.ai/api/v1"
    )

    api_key_env = (
        "OPENROUTER_API_KEY"
    )

    def __init__(self, model: str | None = None) -> None:
        super().__init__(
            provider_name="openrouter",
            model=model,
        )

# =============================================================================
# Together AI Provider
# =============================================================================

class TogetherProvider(OpenAICompatibleProvider):
    """
    Together AI provider integration.
    """

    api_base_url = (
        "https://api.together.xyz/v1"
    )

    api_key_env = (
        "TOGETHER_API_KEY"
    )

    def __init__(
        self,
        model: str | None = None,
    ) -> None:

        super().__init__(
            provider_name="together",
            model=model,
        )

# =============================================================================
# Fireworks Provider
# =============================================================================

class FireworksProvider(OpenAICompatibleProvider):
    """
    Fireworks AI provider integration.
    """

    api_base_url = (
        "https://api.fireworks.ai/inference/v1"
    )

    api_key_env = (
        "FIREWORKS_API_KEY"
    )

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        super().__init__(
            provider_name="fireworks",
            model=model,
        )

# =============================================================================
# xAI Provider
# =============================================================================

class XAIProvider(
    OpenAICompatibleProvider
):

    api_base_url = (
        "https://api.x.ai/v1"
    )

    api_key_env = (
        "XAI_API_KEY"
    )

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        super().__init__(
            provider_name="xai",
            model=model,
        )

# =============================================================================
# Gemini Provider
# =============================================================================

class GeminiProvider(AIProvider):
    """
    Google Gemini provider integration.
    This provider uses the official Google Generative AI SDK
    to send prompts to Gemini models.

    Supported Features:
        - Chat completion
        - Large context windows
        - Streaming support (future extension)
        - Multimodal capability (future extension)

    Environment Variables:
        GEMINI_API_KEY
    Example:
        provider = GeminiProvider(
            model="gemini-2.5-pro",
        )

        response = provider.send(
            "Explain quantum computing."
        )
    """

    def __init__(
        self,
        model: str | None = None,
    ) -> None:
        super().__init__(
            provider_name="gemini",
            model=model
        )

    def _send_impl(self, prompt: str) -> str:
        """
        Execute Gemini API request.

        Args:
            prompt:
                User input prompt.

        Returns:
            AI-generated response text.

        Raises:
            ProviderConfigurationError:
                If SDK or API key is missing.
            ResponseValidationError:
                If Gemini response format is invalid.
        """

        # ---------------------------------------------------------------------
        # SDK Import
        # ---------------------------------------------------------------------
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install Google Generative AI SDK:\n"
                "pip install google-generativeai"
            ) from exc

        # ---------------------------------------------------------------------
        # API Key Validation
        # ---------------------------------------------------------------------

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:
            raise ProviderConfigurationError(
                "GEMINI_API_KEY not set"
            )

        # ---------------------------------------------------------------------
        # Configure SDK
        # ---------------------------------------------------------------------

        genai.configure(
            api_key=api_key
        )

        # ---------------------------------------------------------------------
        # Create Model Client
        # ---------------------------------------------------------------------

        try:
            model = genai.GenerativeModel(
                self.model
            )

        except Exception as exc:
            raise ProviderConfigurationError(
                f"Invalid Gemini model: {self.model}"
            ) from exc

        # ---------------------------------------------------------------------
        # Execute Request
        # ---------------------------------------------------------------------
        try:
            response = model.generate_content(
                prompt
            )

        except Exception as exc:
            raise ProviderRequestError(
                f"Gemini request failed: {exc}"
            ) from exc

        # ---------------------------------------------------------------------
        # Extract Response Text
        # ---------------------------------------------------------------------

        content = getattr(response, "text", None)

        if (
            not content
            or not isinstance(content, str)
        ):

            raise ResponseValidationError(
                "Invalid Gemini response"
            )

        # ---------------------------------------------------------------------
        # Token Usage Tracking (Best Effort)
        # ---------------------------------------------------------------------

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if usage:
            self.metrics.total_prompt_tokens += (
                getattr(
                    usage,
                    "prompt_token_count",
                    0,
                )
            )
            self.metrics.total_completion_tokens += (
                getattr(
                    usage,
                    "candidates_token_count",
                    0,
                )
            )
        return content.strip()

class HTTPProviderMixin:
     """
     Mixin for providers that use direct HTTP requests.
     Provides common HTTP request logic and error handling.
     """

     def _send_impl(self, prompt: str) -> str:
         """
         Execute HTTP request to provider API.

         This method should be called by subclasses that implement
         direct HTTP interactions instead of SDK usage.
         """

         raise NotImplementedError(
             "HTTPProviderMixin requires _send_impl implementation"
         )
class OpenAICompatibleProvider:
    """
    Base class for OpenAI-compatible providers.

    This class provides a standardized implementation for providers
    that support the OpenAI-compatible chat completions API.
    It handles SDK integration, request execution, response parsing,
    and usage tracking in a consistent manner.

    Subclasses must define:
        - api_base_url: Provider API base URL.
        - api_key_env: Environment variable name for API key.

    Supported Providers:
        - Perplexity
class NativeSDKProvider:

# =============================================================================
# OpenAI-Compatible Provider Base
# =============================================================================

class OpenAICompatibleProvider(AIProvider):

    """
    Generic provider for OpenAI-compatible APIs.
    This base class supports providers implementing the
    OpenAI-compatible chat completions interface.
    Supported Providers:
        - Perplexity
        - DeepSeek
        - xAI
        - Groq
        - Together AI
        - OpenRouter
        - Fireworks AI
    Subclasses must define:
        provider_name:
            Registry provider identifier.
        api_base_url:
            Provider API base URL.
        api_key_env:
            Environment variable name containing API key.
    """

    api_base_url: str = ""
    api_key_env: str = ""

    def __init__(
        self,
        provider_name: str,
        model: str | None = None,
    ) -> None:
        super().__init__(
            provider_name=provider_name,
            model=model,
        )

    def _send_impl(self, prompt: str) -> str:
        """
        Execute OpenAI-compatible request.
        """

        # ---------------------------------------------------------------------
        # SDK Import
        # ---------------------------------------------------------------------

        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install OpenAI SDK:\n"
                "pip install openai"
            ) from exc

        # ---------------------------------------------------------------------
        # API Key
        # ---------------------------------------------------------------------

        api_key = os.getenv(
            self.api_key_env
        )

        if not api_key:
            raise ProviderConfigurationError(
                f"{self.api_key_env} not set"
            )

        # ---------------------------------------------------------------------
        # Client
        # ---------------------------------------------------------------------

        client = OpenAI(
            api_key=api_key,
            base_url=self.api_base_url,
        )

        # ---------------------------------------------------------------------
        # Request
        # ---------------------------------------------------------------------

        try:
            response = (
                client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    timeout=self.timeout,
                    max_tokens=2048,
                )
            )

        except Exception as exc:
            raise ProviderRequestError(
                f"{self.provider_name} request failed: {exc}"
            ) from exc

        # ---------------------------------------------------------------------
        # Usage Metrics
        # ---------------------------------------------------------------------

        usage = getattr(
            response,
            "usage",
            None,
        )
        if usage:
            self.metrics.total_prompt_tokens += (
                getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
            )
            self.metrics.total_completion_tokens += (
                getattr(
                    usage,
                    "completion_tokens",
                    0,
                )
            )
        # ---------------------------------------------------------------------
        # Response Extraction
        # ---------------------------------------------------------------------
        try:
            content = (
                response
                .choices[0]
                .message
                .content
            )
        except Exception as exc:
            raise ResponseValidationError(
                "Invalid response structure"
            ) from exc
        if (
            not content
            or not isinstance(content, str)
        ):
            raise ResponseValidationError(
                "Empty response"
            )
        return content.strip()

AVAILABLE_MODELS: dict[str, list[str]] = {
    provider: metadata.supported_models
    for provider, metadata in PROVIDERS.items()
}

# =============================================================================
# Provider Factory
# =============================================================================

PROVIDER_MAP: dict[str, type[AIProvider]] = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "xai": XAIProvider,
    "deepseek": DeepSeekProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "together": TogetherProvider,
    "fireworks": FireworksProvider,
    "gemini": GeminiProvider,
    "perplexity": PerplexityProvider,
    "deepseek": DeepSeekProvider,
    "xai": XAIProvider,
}


def build_provider(
    name: str,
    model: str | None = None,
) -> AIProvider:
    """
    Build provider instance.
    Args:
        name:
            Provider registry identifier.

        model:
            Optional model override.

    Returns:
        Initialized provider instance.
    """

    normalized_name = name.lower()
    try:
        provider_class = (
            PROVIDER_MAP[normalized_name]
        )
    except KeyError as exc:
        raise ProviderConfigurationError(
            f"Unknown provider '{name}'"
        ) from exc

    return provider_class(model=model)


# =============================================================================

# Public API

# =============================================================================

def ask(
    provider: str,
    prompt: str,
    model: str | None = None,
) -> str:
    """
    High-level AI request interface.
    This function validates inputs,
    initializes the requested provider,
    executes the AI request,
    and safely handles failures.

    Args:

        provider:
            Registered provider name.

        prompt:
            User prompt text.

        model:
            Optional model override.

    Returns:
        AI-generated response text.

    Raises:
        No exceptions are propagated.
        All errors are converted into
        structured error responses.

    """
    # -------------------------------------------------------------------------
    # Provider Validation
    # -------------------------------------------------------------------------

    if not isinstance(provider, str):
        return "[ERROR] provider must be string"

    provider = provider.strip().lower()
    if not provider:
        return "[ERROR] provider is empty"
    if provider not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS.keys()))
        return (
            "[ERROR] Invalid provider "
            f"'{provider}'. "
            f"Available providers: {available}"
        )

    # -------------------------------------------------------------------------
    # Prompt Validation
    # -------------------------------------------------------------------------

    if not isinstance(prompt, str):
        return "[ERROR] prompt must be string"
    prompt = prompt.strip()

    if not prompt:
        return "[ERROR] Invalid prompt"

    # -------------------------------------------------------------------------
    # Model Validation
    # -------------------------------------------------------------------------

    if model is not None:
        if not isinstance(model, str):
            return "[ERROR] model must be string"
        model = model.strip()

        if not model:
            return "[ERROR] model is empty"

        supported_models = (
            PROVIDERS[provider]
            .supported_models
        )

        if model not in supported_models:
            supported = ", ".join(
                supported_models
            )

            return (
                "[ERROR] Invalid model "
                f"'{model}' for provider "
                f"'{provider}'. "
                f"Supported models: {supported}"
            )

    # -------------------------------------------------------------------------
    # Provider Execution
    # -------------------------------------------------------------------------

    try:
        ai_provider = build_provider(
            name=provider,
            model=model,
        )
        response = ai_provider.send(prompt)

        # ---------------------------------------------------------------------
        # Final Response Validation
        # ---------------------------------------------------------------------

        if not isinstance(response, str):
            logger.error(
                "invalid_response_type "
                "provider=%s "
                "type=%s",
                provider,
                type(response).__name__,
            )
            return "[ERROR] Invalid response type"
        response = response.strip()

        if not response:
            return "[ERROR] Empty response"
        return response

    # -------------------------------------------------------------------------
    # Known AI Gateway Errors
    # -------------------------------------------------------------------------

    except AIProviderError as exc:
        logger.error(
            "ai_provider_error "
            "provider=%s "
            "error=%s",
            provider,
            exc,
        )
        return f"[ERROR] {exc}"

    # -------------------------------------------------------------------------
    # Unexpected Errors
    # -------------------------------------------------------------------------
    except Exception as exc:
        logger.exception(
            "unexpected_ask_failure "
            "provider=%s "
            "model=%s "
            "error=%s",
            provider,
            model,
            exc,
        )
        return (
            "[ERROR] Unexpected internal error. "
            "Check logs for details."
        )

# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.
    """

    parser = argparse.ArgumentParser(
        prog="ai_chat",
        description=(
            "Enterprise AI Gateway CLI"
        ),
    )

    parser.add_argument(
        "-p",
        "--provider",
        default="echo",
        help="Provider name",
    )

    parser.add_argument(
        "-m",
        "--prompt",
        help="Prompt text",
    )

    parser.add_argument(
        "-M",
        "--model",
        help="Model override",
    )

    return parser.parse_args(argv)


# =============================================================================
# Main
# =============================================================================


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point.
    """
    args = parse_args(argv)
    prompt = args.prompt

    if not prompt:
        prompt = sys.stdin.read().strip()

    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2

    response = ask(
        provider=args.provider,
        prompt=prompt,
        model=args.model,
    )
    print(response)
    return 0


# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "ask",
    "PROVIDERS",
    "AIProvider",
    "RetryEngine",
    "build_provider",
    "OpenAIProvider",
    "ProviderMetadata",
    "ModelQualityMetrics",
    "HallucinationDetector"
]

# =============================================================================
# Entrypoint
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(main())