from typing import Any
 
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers import registry
from ai_cli.providers.base import BaseProvider
 
 
class AutoProvider(BaseProvider):
    """Fallback provider that tries registered providers in order."""
 
    def __init__(
            self,
            fallback_order: list[str] | None = None,
            **_kwargs: Any
            ):
        super().__init__(**_kwargs)
        registry.ensure_initialized()
 
        default_order = [
            "openai",
            "gemini",
            "deepseek",
            "perplexity",
            "xai",
            "cohere",
            "zai",
            "echo",
        ]
 
        available = set(registry.PROVIDER_MAP.keys())
        base = fallback_order or default_order
 
        self.fallback_order = [p for p in base if p in available]
 
        registry.ensure_initialized()
        self.provider_map = registry.PROVIDER_MAP
 
    def send(
            self,
            prompt: str,
            **_kwargs: Any
            ) -> str:
        errors: list[str] = []
        registry.ensure_initialized()
 
        for provider_name in self.fallback_order:
            provider_cls = self.provider_map.get(provider_name)
 
            if provider_cls is None:
                errors.append(f"{provider_name}: not found")
                continue
 
            # ---- SAFE INSTANTIATION ----
            try:
                provider = provider_cls()
            except Exception as exc:
                msg = str(exc).lower()
 
                # skip known missing-config providers cleanly
                if "api key" in msg or "apikey" in msg or "required" in msg:
                    errors.append(f"{provider_name}: skipped (missing config)")
                    continue
 
                errors.append(f"{provider_name}: init failed ({exc})")
                continue
 
            # ---- EXECUTION ----
            try:
                print(f"Trying provider: {provider_name}")
                result = provider.send(prompt)
                print(f"Provider {provider_name} returned: {result!r}")
                return result
 
            except Exception as exc:
                print(f"Provider {provider_name} failed: {exc}")
                msg = str(exc).lower()
 
                if any(
                    k in msg
                    for k in ["401", "403", "quota", "unauthorized", "invalid"]
                ):
                    errors.append(f"{provider_name}: skipped ({exc})")
                    continue
 
                errors.append(f"{provider_name}: failed ({exc})")
                continue
 
        raise ProviderRequestError(
            "Auto fallback exhausted. Errors:\n" + "\n".join(errors)
        )
 
    def ask(
            self,
            prompt: str,
            **_kwargs: Any
            ) -> str:
        return self.send(prompt)