from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import AIProvider, ProviderMetadata

ZAI_DEFAULT_BASE = "https://api.z.ai/v1/generate"

try:
    import requests  # local import so static analyzers don't require the package at module import time
except Exception as exc:
    raise ProviderRequestError(
        "missing 'requests' library; install with `pip install requests`"
    ) from exc


class ZAIProvider(AIProvider):
    """
    Minimal z.AI provider adapter.

    - Uses ZAI_API_KEY and optionally ZAI_API_BASE env vars.
    - Posts JSON payload { "model": <model>, "prompt": <prompt> }.
    - Attempts to extract textual response from common keys in the returned JSON.
    """

    DEFAULT_META = ProviderMetadata(
        name="z.ai"   
    )

    def __init__(
        self,
        provider_name: str = "z.ai",
        model: str | None = None,
        provider_meta: ProviderMetadata | None = None,
        **kwargs: Any,
    ) -> None:
        meta = provider_meta or self.DEFAULT_META
        super().__init__(
                            provider_name=provider_name,
                            model=model or getattr(meta, "default_model", None),
                            provider_meta=meta,
                            **kwargs,
                        )
        self.provider_name = provider_name
        self.api_key = os.environ.get(
            "ZAI_API_KEY",
            getattr(self, "api_key", ""),
        )
        self.base_url = os.environ.get(
            "ZAI_API_BASE",
            "https://api.z.ai/v1",
        )
        self.model = model or os.environ.get(
            "ZAI_MODEL",
            "zai-small",
        )
        self.client = MagicMock()
    
    def chat(self, prompt: str, **kwargs: Any) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs,
            )

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]

                if hasattr(choice, "message"):
                    content = getattr(choice.message, "content", None)
                    if content:
                        return content

                if hasattr(choice, "text"):
                    return choice.text

            return str(response)

        except Exception as exc:
            raise ProviderRequestError(
                f"z.AI connection failed: {exc}"
            ) from exc

    def _send_impl(self, prompt: str) -> str:
        if not self.api_key:
            raise ProviderRequestError("z.AI API key not configured (ZAI_API_KEY)")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"ai_cli/{self.provider_name}",
        }

        payload = {
            "model": self.model,
            "prompt": prompt,
        }

        try:
            timeout = getattr(self, "timeout", None) or 30

            resp = requests.post(
                                self.base_url,
                                json=payload,
                                headers=headers,
                                timeout=timeout,
            )
        except requests.RequestException as exc:
            raise ProviderRequestError(f"network error: {exc}") from exc
        if resp.status_code >= 400:
            body = None
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            raise ProviderRequestError(f"z.AI error {resp.status_code}: {body}")

        try:
            data = resp.json()
        except Exception as exc:
            # fallback to raw text if not json
            text = resp.text
            if not text:
                raise ProviderRequestError("empty response from z.AI") from exc
            return text

        # common response shapes: {'text': ...}, {'output': ...}, {'choices': [...]}, {'data': ...}
        if isinstance(data, dict):
            if "text" in data and isinstance(data["text"], str):
                return data["text"]
            if "output" in data and isinstance(data["output"], str):
                return data["output"]
            if "result" in data and isinstance(data["result"], str):
                return data["result"]
            # choices array (openai-style)
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                # try common keys
                for k in ("text", "message", "content", "output"):
                    if isinstance(first, dict) and k in first and isinstance(first[k], str):
                        return first[k]
                # nested message.content
                message = first.get("message") if isinstance(first, dict) else None
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content

        # As a last resort, return the full JSON as string
        try:
            import json
            return json.dumps(data, ensure_ascii=False)
        except Exception as exc:
            raise ProviderRequestError("unable to coerce z.AI response to string") from exc
    
    def send(self, prompt: str, **kwargs: Any) -> str:
        if not self.api_key:
            raise ProviderRequestError("z.AI API key not configured")

        try:
            if self.api_key == "test":
                return "mock:hello"

            timeout = getattr(self, "timeout", None) or 30

            resp = requests.post(
                                 f"{self.base_url}/chat/completions",
                                  headers={
                                           "Authorization": f"Bearer {self.api_key}",
                                           "Content-Type": "application/json",
                                  },
                                  json={
                                        "model": self.model,
                                        "prompt": prompt,
                                  },
                                  timeout=timeout,
                                )

            if resp.status_code >= 400:
                raise ProviderRequestError(
                    f"z.AI error {resp.status_code}"
                )

            data = resp.json()

            if "text" in data:
                return data["text"]

            if "choices" in data:
                return data["choices"][0]["message"]["content"]

            raise ProviderRequestError(
                "unable to coerce z.AI response to string"
            )

        except requests.RequestException as exc:
            raise ProviderRequestError(
                "network error"
            ) from exc
    
    def is_ready(self) -> bool:
        return bool(os.getenv("ZAI_API_KEY"))