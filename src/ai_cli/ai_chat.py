from __future__ import annotations
import argparse
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Type

"""
ai_chat.py

CLI to send a prompt to different AI providers.

This module defines a common provider interface and several provider
stubs. Implementations should be extended to call real provider APIs.

Added:
 - --model / -M CLI option to select provider model.
 - per-provider default model map and environment var fallback.
 - providers validate/sanitize prompts via super().send(prompt).
"""

# Per-provider default model names (used if --model not provided and no env var)
PROVIDER_CONFIG = {
    "openai": {
        "default_model": "gpt-5.5",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "max_context": 1000000,
    },

    "anthropic": {
        "default_model": "claude-sonnet-4",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "max_context": 200000,
    },

    "gemini": {
        "default_model": "gemini-3.5-flash",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": True,
        "max_context": 1000000,
    },

    "cohere": {
        "default_model": "command-r-plus",
        "supports_streaming": True,
        "supports_tools": False,
        "supports_vision": False,
        "max_context": 128000,
    },

    "deepseek": {
        "default_model": "deepseek-v3",
        "supports_streaming": True,
        "supports_tools": True,
        "supports_vision": False,
        "max_context": 128000,
    },
}



class AIProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, model: str | None = None) -> None:
        """
        Store an optional model selection. Concrete providers should
        consult self.model when making API calls.
        """
        self.model = model

    @abstractmethod
    def send(self, prompt: str) -> str:
        """Send a prompt to the provider and return the text reply.

        This abstract method performs common, security-minded validation and
        normalization of the incoming prompt. Subclasses must override this
        method to perform the actual provider call. Subclasses that want the
        validated/sanitized prompt can call super().send(prompt) which returns
        the sanitized prompt.

        Validation performed:
          - prompt must be a str and non-empty after stripping
          - rejects NUL bytes
          - enforces maximum length via AI_CLI_MAX_PROMPT env var (default: 10000)
          - removes control characters except newline and tab

        Errors avoid echoing the prompt to prevent leaking sensitive data.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")

        # Normalize whitespace and ensure not empty
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt is empty")

        # Maximum length from environment, fallback to safe default
        try:
            max_len = int(os.getenv("AI_CLI_MAX_PROMPT", "10000"))
        except Exception:
            max_len = 10000
        if max_len <= 0:
            max_len = 10000
        if len(prompt) > max_len:
            raise ValueError("prompt exceeds maximum allowed length")

        # Reject NUL bytes which can be problematic for some sinks/parsers
        if "\x00" in prompt:
            raise ValueError("prompt contains NUL byte")

        # Remove control characters except newline and tab to reduce injection/vector risks
        sanitized = "".join(ch for ch in prompt if ch in ("\n", "\t") or ord(ch) >= 32)

        # Return the sanitized prompt. Subclasses should override this method
        # to perform provider-specific sending and may call super().send(prompt)
        # to obtain the sanitized input.
        return sanitized


class EchoProvider(AIProvider):
    """Simple provider that echoes the prompt back."""

    def send(self, prompt: str) -> str:
        """Return the prompt prefixed by a small message."""
        prompt = super().send(prompt)
        model_info = f" (model={self.model})" if self.model else ""
        return f"(echo{model_info}) {prompt}"


class OpenAIProvider(AIProvider):
    """Provider implementation for OpenAI ChatGPT (example)."""

    def send(self, prompt: str) -> str:
        """
        Send prompt via the openai package.

        Requires the OPENAI_API_KEY environment variable and the
        "openai" package. If the package or key is missing an error
        with guidance is raised.

        The model used is taken from (in order of precedence):
         - provider instance self.model (set by --model)
         - OPENAI_MODEL environment variable
         - default in PROVIDER_DEFAULT_MODELS
        """
        prompt = super().send(prompt)

        try:
            import openai  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime check
            raise RuntimeError(
                "openai package is required for OpenAIProvider. "
                "Install it with `pip install openai`."
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set."
            )

        openai.api_key = api_key

        # Determine effective model
        model = self.model or os.getenv("OPENAI_MODEL") or PROVIDER_DEFAULT_MODELS.get("openai")

        # Use ChatCompletion if available, otherwise try completions API.
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
            )
            # Extract text from the new ChatCompletion shape.
            content = resp.choices[0].message["content"]
            return content
        except AttributeError:
            # Fallback for older completions API (may not apply).
            resp = openai.Completion.create(
                engine=model or os.getenv("OPENAI_MODEL", "text-davinci-003"),
                prompt=prompt,
                max_tokens=1024,
            )
            return resp.choices[0].text


class AnthropicProvider(AIProvider):
    """Stub provider for Anthropic Claude."""

    def send(self, prompt: str) -> str:
        """
        Placeholder for Anthropic Claude call.

        Implement this method using the "anthropic" package or HTTP API
        and the ANTHROPIC_API_KEY environment variable.

        The model can be selected via --model or ANTHROPIC_MODEL env var.
        """
        prompt = super().send(prompt)
        raise NotImplementedError(
            "AnthropicProvider is not implemented. Add API call here."
        )


class GoogleGeminiProvider(AIProvider):
    """Stub provider for Google Gemini."""

    def send(self, prompt: str) -> str:
        """
        Placeholder for Google Gemini call.

        Implement this method using Google's client libraries and the
        appropriate credentials (e.g. GOOGLE_APPLICATION_CREDENTIALS).

        The model can be selected via --model or GEMINI_MODEL env var.
        """
        prompt = super().send(prompt)
        raise NotImplementedError(
            "GoogleGeminiProvider is not implemented. Add API call here."
        )


class GithubCopilotProvider(AIProvider):
    """Stub provider for GitHub Copilot (examples only)."""

    def send(self, prompt: str) -> str:
        """
        Placeholder for GitHub Copilot call.

        Copilot access may require special arrangements and is not
        generally available via a simple public API. Implement as needed.
        """
        prompt = super().send(prompt)
        raise NotImplementedError(
            "GithubCopilotProvider is not implemented. Add API call here."
        )


class DeepseekProvider(AIProvider):
    """Stub provider for Deepseek."""

    def send(self, prompt: str) -> str:
        """
        Placeholder for Deepseek API call.

        Implement actual HTTP or SDK call and use DEEPSEEK_API_KEY.
        """
        prompt = super().send(prompt)
        raise NotImplementedError(
            "DeepseekProvider is not implemented. Add API call here."
        )


class GrokProvider(AIProvider):
    """Stub provider for Grok (examples only)."""

    def send(self, prompt: str) -> str:
        """
        Placeholder for Grok implementation.

        Implement actual integration if an API is available to you.
        """
        prompt = super().send(prompt)
        raise NotImplementedError(
            "GrokProvider is not implemented. Add API call here."
        )
    
class CohereProvider(AIProvider):
    @property
    def capabilities(self) -> dict:
        return {
            "streaming": True,
            "tools": False,
            "vision": False,
            "embeddings": True,
            "async": False,
        }

    def send(self, prompt: str) -> str:
        import cohere
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise NotImplementedError(
                "COHERE_API_KEY is not implemented. Add API call here."
            )

        client = cohere.Client(api_key)
        response = client.chat(
            model=self.model,
            message=prompt,
        )
        return response.text

PROVIDER_DEFAULT_MODELS: Dict[str, str] = {
    # Common generally-available model names
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-4",
    "gemini": "gemini-3.5-flash",
    "copilot": "copilot-next",
    "deepseek": "deepseek-v3",
    "grok": "grok-1.5",
    "cohere": "command-r-plus",
    "echo": ""
}


PROVIDER_MAP: Dict[str, Type[AIProvider]] = {
    "echo": EchoProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GoogleGeminiProvider,
    "copilot": GithubCopilotProvider,
    "deepseek": DeepseekProvider,
    "grok": GrokProvider,
    "cohere": CohereProvider,
}


def build_provider(name: str, model: str | None = None) -> AIProvider:
    """
    Create a provider instance from a provider name.

    The effective model is chosen in this order:
     - explicit model argument (from --model)
     - environment variable <PROVIDER_UPPER>_MODEL (e.g. OPENAI_MODEL)
     - PROVIDER_DEFAULT_MODELS mapping

    Raises KeyError if the provider name is unknown.
    """
    try:
        cls = PROVIDER_MAP[name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown provider '{name}'. Valid: {', '.join(PROVIDER_MAP)}"
        ) from exc

    env_model = os.getenv(f"{name.upper()}_MODEL")
    effective_model = model or env_model or PROVIDER_DEFAULT_MODELS.get(name.lower())
    return cls(effective_model)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Accepts --provider and --prompt. If prompt is omitted the prompt is
    read from stdin. Use --model to select the model for the chosen provider.
    """
    parser = argparse.ArgumentParser(
        prog="ai_chat",
        description="Send a prompt to an AI provider and print reply.",
    )
    parser.add_argument(
        "-p",
        "--provider",
        default="echo",
        help="Provider name (echo, openai, anthropic, gemini, etc).",
    )
    parser.add_argument(
        "-m",
        "--prompt",
        help="Prompt text. If omitted the prompt is read from stdin.",
    )
    parser.add_argument(
        "-M",
        "--model",
        help="Model name to use for the selected provider (overrides env/default).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point for the CLI.

    Returns exit code 0 on success and non-zero on error.
    """
    args = parse_args(argv)
    prompt = args.prompt
    if not prompt:
        # Read entire stdin as the prompt.
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("No prompt provided.", file=sys.stderr)
        return 2

    try:
        provider = build_provider(args.provider, args.model)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 3

    try:
        reply = provider.send(prompt)
    except NotImplementedError as exc:
        print(f"Provider not implemented: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"Error calling provider: {exc}", file=sys.stderr)
        return 5

    print(reply)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI invocation
    raise SystemExit(main())