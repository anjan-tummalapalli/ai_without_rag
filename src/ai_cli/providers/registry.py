from __future__ import annotations
import importlib
import pkgutil
from typing import Dict, Type

from ai_cli.providers.base import AIProvider, ProviderMetadata, EchoProvider

# Attempt to import the optional GeminiProvider dynamically to avoid static import
# resolution errors in environments where the package isn't available.
GeminiProvider = None
try:
    module = importlib.import_module("ai_cli.providers.gemini")
    GeminiProvider = getattr(module, "GeminiProvider", None)
except Exception:
    # If the gemini provider isn't installed or importable, keep a None placeholder
    GeminiProvider = None

PROVIDERS: Dict[str, ProviderMetadata] = {
    "openai": ProviderMetadata(
        name="OpenAI",
        default_model="gpt-5.5",
        supported_models=[
            "gpt-5.5",
            "gpt-4o-realtime-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo-2025",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        cost_per_1k_tokens=0.01,
        avg_latency_ms=800,
    ),
    "perplexity": ProviderMetadata(
        name="Perplexity AI",
        default_model="sonar-pro-v2",
        supported_models=["sonar-pro-v2", "sonar-pro", "sonar"],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.009,
        avg_latency_ms=850,
    ),
    "deepseek": ProviderMetadata(
        name="DeepSeek",
        default_model="deepseek-chat-v2",
        supported_models=[
            "deepseek-chat-v2",
            "deepseek-coder-v2",
            "deepseek-reasoner-v2",
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
        default_model="llama-3.5-70b",
        supported_models=["llama-3.5-70b", "mixtral-8x7b-v2"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.002,
        avg_latency_ms=300,
    ),
    "openrouter": ProviderMetadata(
        name="OpenRouter",
        default_model="openai/gpt-4o-realtime",
        supported_models=[
            "openai/gpt-4o-realtime",
            "anthropic/claude-4-1",
            "google/gemini-3-pro",
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
        default_model="meta-llama/Llama-3-70b-chat-v2",
        supported_models=[
            "meta-llama/Llama-3-70b-chat-v2",
            "mistralai/Mixtral-8x7B-Instruct-v1.1",
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
        default_model="accounts/fireworks/models/llama-v3p2-70b-instruct",
        supported_models=["accounts/fireworks/models/llama-v3p2-70b-instruct"],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=128_000,
        cost_per_1k_tokens=0.004,
        avg_latency_ms=550,
    ),
    "xai": ProviderMetadata(
        name="xAI Grok",
        default_model="grok-4",
        supported_models=["grok-4", "grok-3-mini"],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=128_000,
        cost_per_1k_tokens=0.011,
        avg_latency_ms=750,
    ),
    "gemini": ProviderMetadata(
        name="Google Gemini",
        default_model="gemini-3-pro",
        supported_models=[
            "gemini-3-pro",
            "gemini-2.6",
            "gemini-mini-2",
        ],
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        max_context=1_000_000,
        cost_per_1k_tokens=0.012,
        avg_latency_ms=600,
    ),
    "cohere": ProviderMetadata(
        name="Cohere",
        default_model="command-xlarge-stable",
        supported_models=[
            "command-xlarge-stable",
            "command-nightly",
            "embed-english-v3.0",
        ],
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        max_context=4_096,
        cost_per_1k_tokens=0.007,
        avg_latency_ms=400,
    ),
}

# The actual mapping is populated at runtime
PROVIDER_MAP: Dict[str, Type[AIProvider]] = {"echo": EchoProvider}

def register_provider(name: str, provider_cls: Type[AIProvider], metadata: ProviderMetadata) -> None:
    """Register a provider class under a name and ensure metadata exists."""
    PROVIDER_MAP[name.lower()] = provider_cls
    PROVIDERS.setdefault(name.lower(), metadata)

# Register Google alias for Gemini if the Gemini provider is available
if GeminiProvider is not None:
    register_provider("google", GeminiProvider, PROVIDERS["gemini"])


def build_provider(name: str, model: str | None = None) -> AIProvider:
    """
    Registry and factory for AI provider implementations.

    This module exposes build_provider(name: str, model: str | None = None) -> AIProvider,
    a small factory function that constructs and returns a concrete AIProvider instance
    based on a provider name. The function performs the following high-level steps:

    - Normalizes the provided name to lower-case.
    - Handles the special provider name "auto" by dynamically importing and returning
        an AutoProvider initialized with the optional model argument.
    - For other names, looks up the provider class in the PROVIDER_MAP mapping using
        the normalized name and instantiates it with the model keyword argument.
    - If the name is not found in PROVIDER_MAP, raises ProviderConfigurationError.

    Parameters
    - name (str): The provider identifier (case-insensitive). Special value "auto"
        triggers automatic selection behavior.
    - model (str | None): Optional model identifier to pass to the provider on
        construction. May be None to request provider defaults.

    Returns
    - AIProvider: An initialized provider instance ready for use.

    Raises
    - ProviderConfigurationError: If the provided name is unknown (not "auto" and
        not present in PROVIDER_MAP).
    - ImportError / ModuleNotFoundError: If dynamic import of AutoProvider fails.
    - TypeError / Exception: If a provider class in PROVIDER_MAP cannot be
        instantiated with the given arguments.

    Notes
    - The provider classes stored in PROVIDER_MAP are expected to accept a
        'model' keyword argument in their constructor (or otherwise handle the
        provided signature).
    - This function keeps import scope minimal by importing AutoProvider only when
        the "auto" option is requested, avoiding unnecessary imports at module load
        time.

    End result: a concrete, configured AIProvider instance corresponding to the
    requested provider name (or an error if the provider cannot be resolved).
    """
    """Factory that constructs a provider instance by name."""
    normalized_name = name.lower()

    # Auto logic
    if normalized_name == "auto":
        from ai_cli.providers.auto_provider import AutoProvider

        return AutoProvider(model=model)

    try:
        provider_class = PROVIDER_MAP[normalized_name]
    except KeyError as exc:
        from ai_cli.core.exceptions import ProviderConfigurationError

        raise ProviderConfigurationError(f"Unknown provider '{name}'") from exc
    return provider_class(model=model)


def load_plugins() -> None:
    """Load providers from the plugins directory."""
    import ai_cli.plugins as plugins

    for _, module_name, _ in pkgutil.iter_modules(
        plugins.__path__, plugins.__name__ + "."
    ):
        try:
            importlib.import_module(module_name)
        except Exception as e:
            import logging

            logging.getLogger("ai_gateway").warning(
                f"Failed to load plugin {module_name}: {e}"
            )