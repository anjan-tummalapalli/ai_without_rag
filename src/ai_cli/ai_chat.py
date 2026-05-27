# Lazily re-export selected names from submodules to avoid import-time errors
# and to provide clearer runtime error messages if dependencies are missing.

from typing import TYPE_CHECKING
import importlib

# For static type checkers, import the real symbols so tools like mypy/IDE work.
if TYPE_CHECKING:
    from ai_cli.core.api import ask  # type: ignore
    from ai_cli.providers.registry import PROVIDERS, build_provider  # type: ignore
    from ai_cli.providers.base import AIProvider, ProviderMetadata  # type: ignore
    from ai_cli.core.exceptions import AIProviderError  # type: ignore
    from ai_cli.core.resilience import RetryEngine  # type: ignore
    from ai_cli.utils.validation import HallucinationDetector  # type: ignore
    from ai_cli.telemetry.monitoring import ModelQualityMetrics  # type: ignore

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
