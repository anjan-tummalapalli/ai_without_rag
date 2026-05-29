"""
ai_cli.ai_chat

Lazily re-export selected names from submodules to:
- avoid import-time failures when optional dependencies are missing,
- provide clearer runtime error messages when imports fail,
- and reduce startup cost by deferring work until attributes are actually used.

Recent changes and purpose:
- AVAILABLE_MODELS is now constructed lazily as a special-case in __getattr__
    so we don't import the full PROVIDERS registry at module import time.
- All other exported symbols are defined in _lazy_imports and imported on
    first access. Results are cached on the module globals to avoid repeated
    imports.
- Import failures are converted to RuntimeError with explanatory messages to
    make diagnosing dependency issues easier at runtime.
- TYPE_CHECKING imports remain to provide correct behavior for type checkers
    and IDEs without affecting runtime lazy loading.

This module exposes a stable public surface via __all__ and implements
__getattr__ and __dir__ to support lazy attribute access and discovery.
"""

from typing import TYPE_CHECKING
import importlib
from ai_cli.rag.pipeline import RAGPipeline

# For static type checkers, import the real symbols so tools like mypy/IDE work.
if TYPE_CHECKING:
        from ai_cli.core.api import ask  # type: ignore
        from ai_cli.providers.registry import PROVIDERS, build_provider  # type: ignore
        from ai_cli.providers.base import AIProvider, ProviderMetadata  # type: ignore
        from ai_cli.core.exceptions import AIProviderError  # type: ignore
        from ai_cli.core.resilience import RetryEngine  # type: ignore
        from ai_cli.utils.validation import HallucinationDetector  # type: ignore
        from ai_cli.telemetry.monitoring import ModelQualityMetrics  # type: ignore
        from ai_cli.core.prompt_corrector import PromptCorrector, prompt_corrector  # type: ignore

# Compute AVAILABLE_MODELS lazily to avoid importing PROVIDERS at module import time.
# It will be constructed when requested via __getattr__.

# Mapping of exported names to (module, attribute) for lazy import.
_lazy_imports = {
        "ask": ("ai_cli.core.api", "ask"),
        "PROVIDERS": ("ai_cli.providers.registry", "PROVIDERS"),
        "build_provider": ("ai_cli.providers.registry", "build_provider"),
        "AIProvider": ("ai_cli.providers.base", "AIProvider"),
        "ProviderMetadata": ("ai_cli.providers.base", "ProviderMetadata"),
        "AIProviderError": ("ai_cli.core.exceptions", "AIProviderError"),
        "RetryEngine": ("ai_cli.core.resilience", "RetryEngine"),
        "HallucinationDetector": ("ai_cli.utils.validation", "HallucinationDetector"),
        "ModelQualityMetrics": ("ai_cli.telemetry.monitoring", "ModelQualityMetrics"),
        "PromptCorrector": ("ai_cli.core.prompt_corrector", "PromptCorrector"),
        "prompt_corrector": ("ai_cli.core.prompt_corrector", "prompt_corrector"),
}


__all__ = [
        "ask",
        "PROVIDERS",
        "AIProvider",
        "RetryEngine",
        "build_provider",
        "ProviderMetadata",
        "ModelQualityMetrics",
        "HallucinationDetector",
        "AIProviderError",
        "AVAILABLE_MODELS",
        "PromptCorrector",
        "prompt_corrector",
]


def __getattr__(name: str):
        """
        Lazily import and return attributes defined in _lazy_imports.

        Raises a RuntimeError with a clear message if the import fails,
        and AttributeError if the name is not exported.
        """
        # Special-case: construct AVAILABLE_MODELS lazily from PROVIDERS to avoid import-time access.
        if name == "AVAILABLE_MODELS":
                try:
                        registry = importlib.import_module("ai_cli.providers.registry")
                        providers = getattr(registry, "PROVIDERS")
                        result = {
                                provider: meta.supported_models
                                for provider, meta in providers.items()
                        }
                        globals()[name] = result
                        return result
                except Exception as exc:
                        raise RuntimeError(
                                f"Failed to build '{name}' from 'ai_cli.providers.registry': {exc}"
                        ) from exc

        if name not in _lazy_imports:
                raise AttributeError(f"module {__name__} has no attribute {name!r}")

        module_name, attr_name = _lazy_imports[name]
        try:
                module = importlib.import_module(module_name)
                value = getattr(module, attr_name)
                # Cache on the module so subsequent accesses don't re-import.
                globals()[name] = value
                return value
        except Exception as exc:  # broad catch to transform import errors into clearer messages
                raise RuntimeError(
                        f"Failed to import '{attr_name}' from '{module_name}': {exc}"
                ) from exc
def __dir__():
        # Include the lazy exports in dir()
        return sorted(list(globals().keys()) + list(_lazy_imports.keys()))

rag = RAGPipeline()
context = rag.retrieve_context(prompt)
enhanced_prompt = f"""
Use the following context to answer.

Context:
{context}

Question:
{prompt}
"""

response = provider.ask(
    prompt=enhanced_prompt,
)