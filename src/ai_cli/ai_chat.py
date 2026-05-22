from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
import uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Type

AVAILABLE_MODELS = {
    "openai": [
        "gpt-5.5",
        "gpt-4.1",
    ],
    "gemini": [
        "gemini-2.5-pro",
    ],
}

PROVIDERS = {
    "openai": "OpenAI",
    "gemini": "Google Gemini",
}


def ask(
    provider: str,
    prompt: str,
    model: str | None = None,
) -> str:
    """
    Basic AI request wrapper.
    """

    if (
        not provider
        or not isinstance(provider, str)
        or provider.strip() not in PROVIDERS
    ):
        return "[ERROR] Invalid provider"

    if (
        not prompt
        or not isinstance(prompt, str)
        or not prompt.strip()
    ):
        return "[ERROR] Invalid prompt"

    provider = provider.strip()
    if (
        model
        and model not in AVAILABLE_MODELS.get(provider, [])
    ):
        return "[ERROR] Invalid model"

    # Temporary mock response for tests
    return "Paris is the capital of France."

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(message)s"
    ),
)
logger = logging.getLogger("ai_gateway")

# -----------------------------------------------------------------------------
# Provider Configuration
# -----------------------------------------------------------------------------

PROVIDER_CONFIG = {
    "openai": {
        "default_model": "gpt-5.5",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "max_context": 1000000,
        "cost_per_1k_tokens": 0.01,
        "avg_latency_ms": 800,
    },
    "anthropic": {
        "default_model": "claude-sonnet-4",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "max_context": 200000,
        "cost_per_1k_tokens": 0.012,
        "avg_latency_ms": 900,
    },
    "gemini": {
        "default_model": "gemini-3.5-flash",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "max_context": 1000000,
        "cost_per_1k_tokens": 0.008,
        "avg_latency_ms": 700,
    },
    "cohere": {
        "default_model": "command-r-plus",
        "supports_streaming": True,
        "supports_tools": False,
        "supports_vision": False,
        "max_context": 128000,
        "cost_per_1k_tokens": 0.006,
        "avg_latency_ms": 600,
    },
    "deepseek": {
        "default_model": "deepseek-v3",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "max_context": 128000,
        "cost_per_1k_tokens": 0.003,
        "avg_latency_ms": 500,
    },
    "grok": {
        "default_model": "grok-2",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "max_context": 128000,
        "cost_per_1k_tokens": 0.011,
        "avg_latency_ms": 750,
    },
    "mistral": {
        "default_model": "mistral-large-latest",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "max_context": 128000,
        "cost_per_1k_tokens": 0.007,
        "avg_latency_ms": 650,
    },
    "perplexity": {
        "default_model": "sonar-pro",
        "supports_streaming": True,
        "supports_tools": False,
        "supports_vision": False,
        "max_context": 128000,
        "cost_per_1k_tokens": 0.009,
        "avg_latency_ms": 850,
    },
}

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class AIProviderError(Exception):
    pass

class PromptValidationError(AIProviderError):
    pass

class ProviderConfigurationError(AIProviderError):
    pass

class ProviderRequestError(AIProviderError):
    pass

class ResponseValidationError(AIProviderError):
    pass

# -----------------------------------------------------------------------------
# Hallucination Detection
# -----------------------------------------------------------------------------


@dataclass
class HallucinationResult:
    score: float
    passed: bool
    reasons: list[str]


class HallucinationDetector:
    SUSPICIOUS_PATTERNS = [
        r"100% accurate",
        r"guaranteed",
        r"always works",
        r"never fails",
        r"trust me",
    ]

    def evaluate(
        self,
        prompt: str,
        response: str,
    ) -> HallucinationResult:

        score = 0.0
        reasons = []

        if len(response.strip()) < 5:
            score += 0.4
            reasons.append("response too short")

        for pattern in self.SUSPICIOUS_PATTERNS:

            if re.search(
                pattern,
                response,
                re.IGNORECASE,
            ):
                score += 0.2
                reasons.append(f"suspicious phrase: {pattern}"
                )

        if "TODO" in response:
            score += 0.3
            reasons.append(f"placeholder content detected")

        score = min(score, 1.0)

        return HallucinationResult(
            score=score,
            passed=score < 0.5,
            reasons=reasons,
        )


# -----------------------------------------------------------------------------
# Response Validation
# -----------------------------------------------------------------------------


class ResponseValidator:
    def validate(
        self,
        response: str,
    ) -> None:

        if not response:
            raise ResponseValidationError(
                "empty response"
            )

        if len(response.strip()) < 5:
            raise ResponseValidationError(
                "response too short"
            )


# -----------------------------------------------------------------------------
# Retry Intelligence
# -----------------------------------------------------------------------------


class RetryEngine:
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> None:

        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def execute(
        self,
        func,
    ):

        last_error = None
        for attempt in range(
            1,
            self.max_attempts + 1,
        ):

            try:
                return func()
            except Exception as exc:
                last_error = exc
                sleep_time = (
                    self.base_delay
                    * (2 ** (attempt - 1))
                )
                jitter = random.uniform(0, 0.5)
                logger.warning(
                    "retry_attempt=%s sleep=%s error=%s",
                    attempt,
                    sleep_time,
                    exc,
                )
                time.sleep(sleep_time + jitter)
        raise last_error


# -----------------------------------------------------------------------------
# Model Quality Metrics
# -----------------------------------------------------------------------------


@dataclass
class ModelQualityMetrics:
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
        if self.requests == 0:
            return 0.0
        return (
            (self.requests - self.failures)
            / self.requests
        )

    @property
    def avg_latency(self) -> float:
        if self.requests == 0:
            return 0.0
        return (
            self.total_latency_seconds
            / self.requests
        )

    @property
    def hallucination_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return (
            self.hallucination_failures
            / self.requests
        )

# -----------------------------------------------------------------------------
# Routing Engine
# -----------------------------------------------------------------------------


class RoutingEngine:
    def select_lowest_latency(
        self,
        providers: list[str],
    ) -> str:
        return min(
            providers,
            key=lambda p: PROVIDER_CONFIG[p]["avg_latency_ms"],
        )

    def select_lowest_cost(
        self,
        providers: list[str],
    ) -> str:

        return min(providers, key=lambda p: PROVIDER_CONFIG[p]["cost_per_1k_tokens"])


# -----------------------------------------------------------------------------
# AI Provider Base Class
# -----------------------------------------------------------------------------


class AIProvider(ABC):
    DEFAULT_TIMEOUT_SECONDS = 60
    DEFAULT_MAX_PROMPT_LENGTH = 10000

    def __init__(
        self,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:

        self.model = model
        self.timeout = (timeout or self.DEFAULT_TIMEOUT_SECONDS)
        self.hallucination_detector = HallucinationDetector()
        self.response_validator = ResponseValidator()
        self.retry_engine = RetryEngine()
        self.trace_id = str(uuid.uuid4())

        self.metrics = ModelQualityMetrics(
            provider=self.__class__.__name__,
            model=model or "unknown",
        )

    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "streaming": False,
            "tools": False,
            "vision": False,
        }

    def validate_prompt(
        self,
        prompt: str,
    ) -> str:

        if not isinstance(prompt, str):
            raise PromptValidationError(
                "prompt must be string"
            )
        prompt = prompt.strip()
        if not prompt:
            raise PromptValidationError(
                "prompt is empty"
            )
        max_len = int(
            os.getenv(
                "AI_CLI_MAX_PROMPT",
                self.DEFAULT_MAX_PROMPT_LENGTH,
            )
        )
        if len(prompt) > max_len:
            raise PromptValidationError("prompt exceeds max length")
        if "\x00" in prompt:
            raise PromptValidationError("prompt contains NUL byte")
        sanitized = "".join(
            ch for ch in prompt
            if ch in ("\n", "\t")
            or ord(ch) >= 32
        )

        return sanitized

    @abstractmethod
    def _send_impl(
        self,
        prompt: str,
    ) -> str:
        pass

    def send(
        self,
        prompt: str,
    ) -> str:

        validated_prompt = self.validate_prompt(prompt)

        start_time = time.monotonic()
        self.metrics.requests += 1
        logger.info(
            "provider_request "
            "provider=%s "
            "model=%s "
            "trace_id=%s",
            self.__class__.__name__,
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

            self.metrics.total_latency_seconds += (duration)

            self.response_validator.validate(response)

            hallucination = (
                self.hallucination_detector
                .evaluate(
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
                    "reasons=%s "
                    "trace_id=%s",
                    self.__class__.__name__,
                    hallucination.score,
                    hallucination.reasons,
                    self.trace_id,
                )
            logger.info(
                "provider_response "
                "provider=%s "
                "latency_seconds=%.3f "
                "trace_id=%s",
                self.__class__.__name__,
                duration,
                self.trace_id,
            )
            return response.strip()

        except Exception as exc:
            self.metrics.failures += 1
            logger.exception(
                "provider_error "
                "provider=%s "
                "trace_id=%s "
                "error=%s",
                self.__class__.__name__,
                self.trace_id,
                exc,
            )
            raise ProviderRequestError(
                f"{self.__class__.__name__} "
                f"request failed: {exc}"
            ) from exc


# -----------------------------------------------------------------------------
# Providers
# -----------------------------------------------------------------------------
class EchoProvider(AIProvider):
    def _send_impl(self, prompt: str) -> str:
        model_info = (
            f" (model={self.model})"
            if self.model else "")
        return f"(echo{model_info}) {prompt}"

class OpenAIProvider(AIProvider):
    @property
    def capabilities(self) -> dict:
        return PROVIDER_CONFIG["openai"]

    def _send_impl(
        self,
        prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install openai package"
            ) from exc

        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        if not api_key:
            raise ProviderConfigurationError(
                "OPENAI_API_KEY not set"
            )

        model = (
            self.model
            or PROVIDER_CONFIG["openai"]["default_model"])
        client = OpenAI(
            api_key=api_key,
        )
        response = (
            client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=2048,
                timeout=self.timeout,
            )
        )

        usage = getattr(response,"usage",None)
        if usage:
            self.metrics.total_prompt_tokens += (
                getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
            )
            self.metrics.total_completion_tokens += (
                getattr(usage, "completion_tokens", 0)
            )
        return (
            response.choices[0].message.content
        )

class CohereProvider(AIProvider):
    @property
    def capabilities(self) -> dict:
        return PROVIDER_CONFIG["cohere"]

    def _send_impl(self, prompt: str) -> str:
        try:
            import cohere
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install cohere package"
            ) from exc

        api_key = os.getenv(
            "COHERE_API_KEY"
        )
        if not api_key:
            raise ProviderConfigurationError(
                "COHERE_API_KEY not set"
            )
        model = (
            self.model
            or PROVIDER_CONFIG["cohere"][
                "default_model"
            ]
        )
        client = cohere.Client(api_key)
        response = client.chat(
            model=model,
            message=prompt,
        )

        return response.text

# -----------------------------------------------------------------------------
# Mistral Provider
# -----------------------------------------------------------------------------

class MistralProvider(AIProvider):

    @property
    def capabilities(self) -> dict:
        return PROVIDER_CONFIG["mistral"]

    def _send_impl(
        self,
        prompt: str,
    ) -> str:

        try:
            from mistralai import Mistral

        except Exception as exc:

            raise ProviderConfigurationError(
                "Install latest mistralai package: "
                "pip install -U mistralai"
            ) from exc

        api_key = os.getenv(
            "MISTRAL_API_KEY"
        )

        if not api_key:

            raise ProviderConfigurationError(
                "MISTRAL_API_KEY not set"
            )

        model = (
            self.model
            or PROVIDER_CONFIG["mistral"][
                "default_model"
            ]
        )

        client = Mistral(
            api_key=api_key,
        )

        response = (
            client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                max_tokens=2048,
            )
        )

        # -------------------------------------------------------------
        # Token Tracking
        # -------------------------------------------------------------

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

        # -------------------------------------------------------------
        # Response Validation
        # -------------------------------------------------------------

        if (
            not response.choices
            or not response.choices[0].message
        ):

            raise ResponseValidationError(
                "invalid mistral response"
            )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not isinstance(content, str):

            raise ResponseValidationError(
                "mistral response invalid"
            )

        return content.strip()
    
# -----------------------------------------------------------------------------
# Perplexity Provider
# -----------------------------------------------------------------------------

class PerplexityProvider(AIProvider):
    API_BASE_URL = ("https://api.perplexity.ai")

    @property
    def capabilities(self) -> dict:
        return PROVIDER_CONFIG[
            "perplexity"
        ]

    def _send_impl(self, prompt: str) -> str:
        try:
            import requests
        except Exception as exc:
            raise ProviderConfigurationError(
                "Install requests package"
            ) from exc

        api_key = os.getenv("PERPLEXITY_API_KEY")

        if not api_key:
            raise ProviderConfigurationError(
                "PERPLEXITY_API_KEY not set"
            )

        model = (
            self.model
            or PROVIDER_CONFIG["perplexity"]["default_model"]
        )

        headers = {
            "Authorization": ("Bearer {api_key}"),
            "Content-Type": ("application/json"),
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an enterprise AI assistant."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": 2048,
            "temperature": 0.2,
        }

        try:
            response = requests.post(
                (
                    f"{self.API_BASE_URL}"
                    "/chat/completions"
                ),
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )

        except requests.RequestException as exc:
            raise ProviderRequestError(
                "Perplexity request failed"
            ) from exc

        # ---------------------------------------------------------------------
        # HTTP Validation
        # ---------------------------------------------------------------------

        if response.status_code >= 400:
            raise ProviderRequestError(
                (
                    "Perplexity API error "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

            )

        try:
            data = response.json()
        except Exception as exc:
            raise ResponseValidationError(
                "Invalid JSON response"
            ) from exc

        # ---------------------------------------------------------------------
        # Token Usage Tracking
        # ---------------------------------------------------------------------

        usage = data.get("usage",{})
        self.metrics.total_prompt_tokens += (
            usage.get("prompt_tokens", 0)
        )

        self.metrics.total_completion_tokens += (
            usage.get("completion_tokens",0)
)

        # ---------------------------------------------------------------------
        # Response Extraction
        # ---------------------------------------------------------------------

        try:
            content = (data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ResponseValidationError(
                    "Invalid Perplexity "
                    "response structure"
            ) from exc
        if (not content or not isinstance(content, str)):
            raise ResponseValidationError("Empty Perplexity response")

        return content.strip()


# -----------------------------------------------------------------------------
# Provider Registry
# -----------------------------------------------------------------------------

PROVIDER_MAP: Dict[str, Type[AIProvider]] = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "cohere": CohereProvider,
    "perplexity": PerplexityProvider,
    "mistal": MistralProvider
}

# -----------------------------------------------------------------------------
# Provider Factory
# -----------------------------------------------------------------------------


def build_provider(name: str, model: str | None = None) -> AIProvider:
    try:
        cls = PROVIDER_MAP[name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown provider '{name}'"
        ) from exc
    env_model = os.getenv(
        f"{name.upper()}_MODEL"
    )

    effective_model = (
        model
        or env_model
        or PROVIDER_CONFIG
        .get(name.lower(), {})
        .get("default_model")
    )
    return cls(effective_model)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai_chat",
        description=(
            "Enterprise AI Gateway CLI"
        ),
    )
    parser.add_argument("-p", "--provider", default="echo")
    parser.add_argument("-m", "--prompt")
    parser.add_argument("-M", "--model")
    return parser.parse_args(argv)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = args.prompt
    if not prompt:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2
    try:
        provider = build_provider(
            args.provider,
            args.model,
        )

        print(provider.send(prompt))
        
        print("\n--- Metrics ---")
        print(f"Requests: {provider.metrics.requests}")

        print(f"Failures: {provider.metrics.failures}")
        
        print(f"Success Rate: {provider.metrics.success_rate:.2%}")
        print(f"Avg Latency: {provider.metrics.avg_latency:.3f}s")
        print(f"Hallucination Rate: {provider.metrics.hallucination_rate:.2%}")
        print(f"Prompt Tokens: {provider.metrics.total_prompt_tokens}")
        print(f"Completion Tokens: {provider.metrics.total_completion_tokens}")
        return 0

    except Exception as exc:
        print(
            f"Fatal error: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())