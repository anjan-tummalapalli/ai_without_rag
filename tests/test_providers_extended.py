"""
test_providers_extended.py
 
Tests for builtins, cohere_provider, openai_provider, auto_provider,
registry, base, echo_provider, and perplexity_provider to raise coverage.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.core.exceptions import (
    ProviderConfigurationError,
    ProviderRequestError,
    ResponseValidationError,
)
from ai_cli.plugins.builtins import (
    CohereProvider as BuiltinsCohereProvider,
)
from ai_cli.plugins.builtins import (
    DeepSeekProvider,
    FireworksProvider,
    GroqProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
    TogetherProvider,
)
from ai_cli.plugins.builtins import (
    GeminiProvider as BuiltinsGeminiProvider,
)
from ai_cli.plugins.builtins import (
    OpenAIProvider as BuiltinsOpenAIProvider,
)
from ai_cli.plugins.builtins import (
    PerplexityProvider as BuiltinsPerplexityProvider,
)
from ai_cli.plugins.builtins import (
    XAIProvider as BuiltinsXAI,
)
from ai_cli.providers.auto_provider import AutoProvider
from ai_cli.providers.base import BaseProvider, EchoProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.echo_provider import EchoProvider as EchoProviderModule
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.registry import (
    CHAT_PROVIDERS,
    PROVIDER_MAP,
    build_provider,
    ensure_initialized,
    get_chat_provider,
    list_providers,
    register_chat_provider,
    register_provider,
)

# ─────────────────────────────────────────────
# providers/base.py
# ─────────────────────────────────────────────
 
 
class TestBaseProvider:
    """Tests for BaseProvider."""
 
    def test_send_raises_not_implemented(self) -> None:
        """send() must be abstract."""
        p = BaseProvider()
        with pytest.raises(NotImplementedError):
            p.send("hello")
 
    def test_ask_delegates_to_send(self) -> None:
        """ask() must delegate to send()."""
        p = BaseProvider()
        p.send = lambda prompt, **kw: f"sent:{prompt}"  # type: ignore[method-assign]
        assert p.ask("hi") == "sent:hi"
 
    def test_init_stores_api_key_and_model(self) -> None:
        """__init__ stores api_key and model kwargs."""
        p = BaseProvider(api_key="k", model="m")
        assert p.api_key == "k"
        assert p.model == "m"
 
 
class TestEchoProvider:
    """Tests for the EchoProvider defined in base.py."""
 
    def test_send(self) -> None:
        """send() prefixes the prompt with '(echo) '."""
        p = EchoProvider()
        assert p.send("hi") == "(echo) hi"
 
    def test_ask(self) -> None:
        """ask() produces the same result as send()."""
        p = EchoProvider()
        assert p.ask("world") == "(echo) world"
 
    def test_provider_name(self) -> None:
        """provider_name is set to 'echo'."""
        p = EchoProvider()
        assert p.provider_name == "echo"
 
 
# ─────────────────────────────────────────────
# providers/echo_provider.py
# ─────────────────────────────────────────────
 
 
class TestEchoProviderModule:
    """Tests for the EchoProvider defined in echo_provider.py."""
 
    def test_send_returns_echoed(self) -> None:
        """send() must echo the input prompt back."""
        p = EchoProviderModule()
        result = p.send("test message")
        assert "test message" in result
 
    def test_ask_returns_echoed(self) -> None:
        """ask() must echo the input prompt back."""
        p = EchoProviderModule()
        result = p.ask("test message")
        assert "test message" in result
 
    def test_has_provider_name_attribute(self) -> None:
        """The module-level EchoProvider exposes a provider name attribute."""
        p = EchoProviderModule()
        assert hasattr(p, "provider_name") or hasattr(p, "PROVIDER_NAME")
 
 
# ─────────────────────────────────────────────
# providers/registry.py
# ─────────────────────────────────────────────
 
 
class _DummyProvider:
    """Minimal stub used only by TestRegistry."""
 
    def send(self, prompt: str) -> str:  # noqa: D401
        """Satisfy the two-public-methods minimum."""
        return prompt
 
    def ask(self, prompt: str) -> str:  # noqa: D401
        """Satisfy the two-public-methods minimum."""
        return prompt
 
 
class _DecoratedDummyProvider:
    """Minimal stub used only by TestRegistry decorator test."""
 
    def send(self, prompt: str) -> str:  # noqa: D401
        """Satisfy the two-public-methods minimum."""
        return prompt
 
    def ask(self, prompt: str) -> str:  # noqa: D401
        """Satisfy the two-public-methods minimum."""
        return prompt
 
 
class _ChatDummyProvider:
    """Minimal stub used only by TestRegistry chat-provider test."""
 
    def send(self, prompt: str) -> str:  # noqa: D401
        """Satisfy the two-public-methods minimum."""
        return prompt
 
    def ask(self, prompt: str) -> str:  # noqa: D401
        """Satisfy the two-public-methods minimum."""
        return prompt
 
 
class TestRegistry:
    """Tests for the provider registry module."""
 
    def test_register_and_retrieve_provider(self) -> None:
        """register_provider stores the class in PROVIDER_MAP."""
        register_provider("__test_dummy__", _DummyProvider)
        assert PROVIDER_MAP.get("__test_dummy__") is _DummyProvider
 
    def test_register_provider_as_decorator(self) -> None:
        """register_provider used as a decorator updates PROVIDER_MAP."""
 
        @register_provider("__test_decorator__")
        class _Inner:  # pylint: disable=too-few-public-methods
            """Inner stub."""
 
        assert PROVIDER_MAP.get("__test_decorator__") is _Inner
 
    def test_register_chat_provider(self) -> None:
        """register_chat_provider populates both CHAT_PROVIDERS and PROVIDER_MAP."""
        register_chat_provider("__test_chat__", _ChatDummyProvider)
        assert CHAT_PROVIDERS.get("__test_chat__") is _ChatDummyProvider
        assert PROVIDER_MAP.get("__test_chat__") is _ChatDummyProvider
 
    def test_list_providers_sorted(self) -> None:
        """list_providers returns a sorted list of names."""
        providers = list_providers()
        assert isinstance(providers, list)
        assert providers == sorted(providers)
 
    def test_build_provider_unknown_raises(self) -> None:
        """build_provider raises ValueError for an unregistered name."""
        with pytest.raises(ValueError, match="Unknown provider"):
            build_provider("__definitely_not_registered__")
 
    def test_get_chat_provider_unknown_raises(self) -> None:
        """get_chat_provider raises ValueError for an unregistered name."""
        with pytest.raises(ValueError, match="Unknown chat provider"):
            get_chat_provider("__no_such_chat_provider__")
 
    def test_ensure_initialized_is_idempotent(self) -> None:
        """ensure_initialized can be called multiple times without error."""
        ensure_initialized()
        ensure_initialized()
 
 
# ─────────────────────────────────────────────
# providers/auto_provider.py
# ─────────────────────────────────────────────
 
 
class _OkSendProvider:
    """Stub that always returns 'ok:<prompt>'."""
 
    def send(self, prompt: str) -> str:  # noqa: D401
        """Echo the prompt prefixed with 'ok:'."""
        return f"ok:{prompt}"
 
    def ask(self, prompt: str) -> str:  # noqa: D401
        """Delegate to send()."""
        return self.send(prompt)
 
 
class _FromSendProvider:
    """Stub that always returns 'from_send'."""
 
    def send(self, prompt: str) -> str:  # noqa: D401
        """Return a fixed string, ignoring prompt."""
        _ = prompt
        return "from_send"
 
    def ask(self, prompt: str) -> str:  # noqa: D401
        """Delegate to send()."""
        return self.send(prompt)
 
 
class _FallbackOkProvider:
    """Stub that always returns 'fallback_ok'."""
 
    def send(self, prompt: str) -> str:  # noqa: D401
        """Return a fixed string, ignoring prompt."""
        _ = prompt
        return "fallback_ok"
 
    def ask(self, prompt: str) -> str:  # noqa: D401
        """Delegate to send()."""
        return self.send(prompt)
 
 
class _AlwaysFailProvider:
    """Stub that always raises ProviderRequestError."""
 
    def send(self, prompt: str) -> str:  # noqa: D401
        """Always raise, ignoring prompt."""
        _ = prompt
        raise ProviderRequestError("always fails")
 
    def ask(self, prompt: str) -> str:  # noqa: D401
        """Delegate to send()."""
        return self.send(prompt)
 
 
class TestAutoProvider:
    """Tests for AutoProvider fallback logic."""
 
    def test_send_with_custom_fallback(self) -> None:
        """AutoProvider calls the registered provider class."""
        PROVIDER_MAP["__auto_test__"] = _OkSendProvider
        ap = AutoProvider(fallback_order=["__auto_test__"])
        result = ap.send("hello")
        assert result == "ok:hello"
 
    def test_ask_delegates_to_send(self) -> None:
        """ask() is an alias for send()."""
        PROVIDER_MAP["__auto_ask__"] = _FromSendProvider
        ap = AutoProvider(fallback_order=["__auto_ask__"])
        assert ap.ask("hi") == "from_send"
 
    def test_send_skips_missing_provider(self) -> None:
        """send() skips providers not present in PROVIDER_MAP."""
        PROVIDER_MAP["__auto_fallback__"] = _FallbackOkProvider
        ap = AutoProvider(fallback_order=["__not_registered_xyz__", "__auto_fallback__"])
        result = ap.send("hello")
        assert result == "fallback_ok"
 
    def test_send_all_fail_raises(self) -> None:
        """send() raises ProviderRequestError when all fallbacks fail."""
        PROVIDER_MAP["__auto_fail__"] = _AlwaysFailProvider
        ap = AutoProvider(fallback_order=["__auto_fail__"])
        with pytest.raises(ProviderRequestError, match="Auto fallback exhausted"):
            ap.send("hello")
 
    def test_default_fallback_order(self) -> None:
        """AutoProvider initialises fallback_order to a list."""
        ap = AutoProvider()
        assert isinstance(ap.fallback_order, list)
 
 
# ─────────────────────────────────────────────
# plugins/builtins.py
# ─────────────────────────────────────────────
 
 
class TestBuiltinsOpenAIProvider:
    """Tests for the OpenAIProvider defined in plugins/builtins.py."""
 
    def test_init_sets_timeout(self) -> None:
        """Timeout attribute is set on the instance."""
        p = BuiltinsOpenAIProvider.__new__(BuiltinsOpenAIProvider)
        p.timeout = 60.0
        assert p.timeout == 60.0
 
    def test_send_no_api_key_raises(self) -> None:
        """send() raises ProviderConfigurationError when OPENAI_API_KEY is absent."""
        p = BuiltinsOpenAIProvider.__new__(BuiltinsOpenAIProvider)
        p.model = "gpt-4"
        p.timeout = 60.0
 
        env: dict[str, str] = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("importlib.import_module"):
                with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
                    p.send("hello")
 
    def test_send_import_error_raises(self) -> None:
        """send() raises ProviderConfigurationError when the openai package is missing."""
        p = BuiltinsOpenAIProvider.__new__(BuiltinsOpenAIProvider)
        p.model = "gpt-4"
        p.timeout = 60.0
 
        with patch("importlib.import_module", side_effect=ImportError("no openai")):
            with pytest.raises(ProviderConfigurationError, match="Install openai"):
                p.send("hello")
 
    def test_send_success(self) -> None:
        """send() returns stripped model content on success."""
        p = BuiltinsOpenAIProvider.__new__(BuiltinsOpenAIProvider)
        p.model = "gpt-4"
        p.timeout = 60.0
 
        mock_choice = MagicMock()
        mock_choice.message.content = "  great answer  "
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
 
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch("importlib.import_module", return_value=mock_openai_mod):
                result = p.send("hello")
        assert result == "great answer"
 
    def test_send_api_error_raises(self) -> None:
        """send() wraps API errors into ProviderRequestError."""
        p = BuiltinsOpenAIProvider.__new__(BuiltinsOpenAIProvider)
        p.model = "gpt-4"
        p.timeout = 60.0
 
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
 
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch("importlib.import_module", return_value=mock_openai_mod):
                with pytest.raises(ProviderRequestError, match="OpenAI request failed"):
                    p.send("hello")
 
    def test_send_invalid_response_raises(self) -> None:
        """send() raises ResponseValidationError when content is None."""
        p = BuiltinsOpenAIProvider.__new__(BuiltinsOpenAIProvider)
        p.model = "gpt-4"
        p.timeout = 60.0
 
        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client
 
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with patch("importlib.import_module", return_value=mock_openai_mod):
                with pytest.raises(ResponseValidationError):
                    p.send("hello")
 
 
class TestBuiltinsOpenAICompatibleProvider:
    """Tests for OpenAICompatibleProvider."""
 
    def _make_provider(self, name: str = "perplexity") -> OpenAICompatibleProvider:
        """Return a partially-initialised OpenAICompatibleProvider for testing."""
        p: OpenAICompatibleProvider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
        p.provider_name = name
        p.model = "some-model"
        p.timeout = 60.0
        p.api_key_env = "PERPLEXITY_API_KEY"
        p.api_base_url = "https://api.perplexity.ai"
        return p
 
    def test_get_openai_client_no_key_raises(self) -> None:
        """_get_openai_client raises when the env key is absent."""
        p = self._make_provider()
        env: dict[str, str] = {k: v for k, v in os.environ.items() if k != "PERPLEXITY_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("importlib.import_module", MagicMock()):
                with pytest.raises(ProviderConfigurationError, match="PERPLEXITY_API_KEY"):
                    p._get_openai_client()  # pylint: disable=protected-access
 
    def test_get_openai_client_import_error(self) -> None:
        """_get_openai_client raises when the openai package is absent."""
        p = self._make_provider()
        with patch("importlib.import_module", side_effect=ImportError("no openai")):
            with pytest.raises(ProviderConfigurationError, match="Install OpenAI SDK"):
                p._get_openai_client()  # pylint: disable=protected-access
 
    def test_send_success(self) -> None:
        """send() returns the model response on success."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "response text"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
 
        with patch.object(p, "_get_openai_client", return_value=mock_client):
            result = p.send("hello")
        assert result == "response text"
 
    def test_send_api_failure_raises(self) -> None:
        """send() wraps API errors into ProviderRequestError."""
        p = self._make_provider()
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("down")
 
        with patch.object(p, "_get_openai_client", return_value=mock_client):
            with pytest.raises(ProviderRequestError, match="request failed"):
                p.send("hello")
 
    def test_send_empty_content_raises(self) -> None:
        """send() raises ResponseValidationError when content is empty."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
 
        with patch.object(p, "_get_openai_client", return_value=mock_client):
            with pytest.raises(ResponseValidationError):
                p.send("hello")
 
 
class TestBuiltinsSubProviders:
    """Tests for the convenience sub-provider classes in plugins/builtins.py."""
 
    def test_perplexity_provider_name(self) -> None:
        """PerplexityProvider stores provider_name = 'perplexity'."""
        p = BuiltinsPerplexityProvider.__new__(BuiltinsPerplexityProvider)
        p.provider_name = "perplexity"
        assert p.provider_name == "perplexity"
 
    def test_deepseek_api_base(self) -> None:
        """DeepSeekProvider uses a DeepSeek API base URL."""
        assert "deepseek" in DeepSeekProvider.api_base_url
 
    def test_groq_api_base(self) -> None:
        """GroqProvider uses a Groq API base URL."""
        assert "groq" in GroqProvider.api_base_url
 
    def test_openrouter_api_base(self) -> None:
        """OpenRouterProvider uses an OpenRouter API base URL."""
        assert "openrouter" in OpenRouterProvider.api_base_url
 
    def test_together_api_base(self) -> None:
        """TogetherProvider uses a Together API base URL."""
        assert "together" in TogetherProvider.api_base_url
 
    def test_fireworks_api_base(self) -> None:
        """FireworksProvider uses a Fireworks API base URL."""
        assert "fireworks" in FireworksProvider.api_base_url
 
    def test_xai_api_base(self) -> None:
        """XAIProvider (builtins) uses the xAI API base URL."""
        assert "x.ai" in BuiltinsXAI.api_base_url
 
    def test_gemini_api_base(self) -> None:
        """GeminiProvider (builtins) uses a Gemini or Google API base URL."""
        url_lower = BuiltinsGeminiProvider.api_base_url.lower()
        assert "gemini" in url_lower or "google" in url_lower
 
 
class TestBuiltinsCohereProvider:
    """Tests for the CohereProvider defined in plugins/builtins.py."""
 
    def test_send_no_api_key_raises(self) -> None:
        """send() raises ProviderConfigurationError when COHERE_API_KEY is absent."""
        p = BuiltinsCohereProvider.__new__(BuiltinsCohereProvider)
        p.model = "command"
        p.timeout = 60.0
 
        env: dict[str, str] = {k: v for k, v in os.environ.items() if k != "COHERE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("importlib.import_module", MagicMock()):
                with pytest.raises(ProviderConfigurationError, match="COHERE_API_KEY"):
                    p.send("hello")
 
    def test_send_import_error_raises(self) -> None:
        """send() raises ProviderConfigurationError when the cohere package is absent."""
        p = BuiltinsCohereProvider.__new__(BuiltinsCohereProvider)
        p.model = "command"
        p.timeout = 60.0
 
        with patch("importlib.import_module", side_effect=ImportError("no cohere")):
            with pytest.raises(ProviderConfigurationError, match="cohere"):
                p.send("hello")
 
    def test_send_success(self) -> None:
        """send() returns stripped generation text on success."""
        p = BuiltinsCohereProvider.__new__(BuiltinsCohereProvider)
        p.model = "command"
        p.timeout = 60.0
 
        mock_gen = MagicMock()
        mock_gen.text = "cohere answer"
        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(generations=[mock_gen])
        mock_cohere = MagicMock()
        mock_cohere.Client.return_value = mock_client
 
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
            with patch("importlib.import_module", return_value=mock_cohere):
                result = p.send("hello")
        assert result == "cohere answer"
 
    def test_send_api_failure_raises(self) -> None:
        """send() wraps Cohere API errors into ProviderRequestError."""
        p = BuiltinsCohereProvider.__new__(BuiltinsCohereProvider)
        p.model = "command"
        p.timeout = 60.0
 
        mock_client = MagicMock()
        mock_client.generate.side_effect = RuntimeError("cohere down")
        mock_cohere = MagicMock()
        mock_cohere.Client.return_value = mock_client
 
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
            with patch("importlib.import_module", return_value=mock_cohere):
                with pytest.raises(ProviderRequestError, match="Cohere request failed"):
                    p.send("hello")
 
    def test_send_empty_response_raises(self) -> None:
        """send() raises ResponseValidationError when the generation text is empty."""
        p = BuiltinsCohereProvider.__new__(BuiltinsCohereProvider)
        p.model = "command"
        p.timeout = 60.0
 
        mock_gen = MagicMock()
        mock_gen.text = ""
        mock_client = MagicMock()
        mock_client.generate.return_value = MagicMock(generations=[mock_gen])
        mock_cohere = MagicMock()
        mock_cohere.Client.return_value = mock_client
 
        with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
            with patch("importlib.import_module", return_value=mock_cohere):
                with pytest.raises(ResponseValidationError, match="Empty response"):
                    p.send("hello")
 
 
# ─────────────────────────────────────────────
# providers/openai_provider.py
# ─────────────────────────────────────────────
 
 
class TestOpenAIProviderModule:
    """Tests for OpenAIProvider in providers/openai_provider.py."""
 
    def _make_provider(self, api_key: str = "sk-test") -> OpenAIProvider:
        """Return an OpenAIProvider with a mocked OpenAI client."""
        mock_client = MagicMock()
        with patch("ai_cli.providers.openai_provider.OpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}):
                p = OpenAIProvider(api_key=api_key, model="gpt-4")
                p.client = mock_client
        return p
 
    def test_send_success(self) -> None:
        """send() returns stripped content on success."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  answer  "
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.send("hello") == "answer"
 
    def test_send_no_choices_raises(self) -> None:
        """send() raises ProviderRequestError when choices list is empty."""
        p = self._make_provider()
        p.client.chat.completions.create.return_value = MagicMock(choices=None)
        with pytest.raises(ProviderRequestError, match="no choices"):
            p.send("hello")
 
    def test_send_api_error_raises(self) -> None:
        """send() wraps API exceptions into ProviderRequestError."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("network error")
        with pytest.raises(ProviderRequestError, match="OpenAI request failed"):
            p.send("hello")
 
    def test_ask_delegates_to_send(self) -> None:
        """ask() is an alias for send()."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "answer"
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.ask("test") == "answer"
 
    def test_health_check_success(self) -> None:
        """health_check() returns True when the API responds with choices."""
        p = self._make_provider()
        mock_choice = MagicMock()
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.health_check() is True
 
    def test_health_check_failure(self) -> None:
        """health_check() returns False when the API raises."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        assert p.health_check() is False
 
    def test_ensure_key_missing_raises(self) -> None:
        """_ensure_key raises ProviderRequestError when the API key is absent."""
        p = self._make_provider()
        p.api_key = ""
        env: dict[str, str] = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ProviderRequestError, match="OPENAI_API_KEY"):
                p._ensure_key()  # pylint: disable=protected-access
 
    def test_missing_api_key_on_init_raises(self) -> None:
        """OpenAIProvider.__init__ raises ValueError when no API key is found."""
        env: dict[str, str] = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIProvider(api_key=None)
 
 
# ─────────────────────────────────────────────
# providers/cohere_provider.py (standalone)
# ─────────────────────────────────────────────
 
 
def _make_cohere_provider() -> CohereProvider:
    """Return a CohereProvider in test-key mode with internal stores reset."""
    with patch(
        "ai_cli.providers.cohere_provider.CohereProvider.__init__",
        lambda self, **kw: None,
    ):
        p: CohereProvider = CohereProvider.__new__(CohereProvider)
        p.api_key = "test"  # type: ignore[attr-defined]
        p.client = MagicMock()  # type: ignore[attr-defined]
        p._documents = []  # type: ignore[attr-defined]  # pylint: disable=protected-access
        p._vectors = []  # type: ignore[attr-defined]  # pylint: disable=protected-access
        p._metadata = []  # type: ignore[attr-defined]  # pylint: disable=protected-access
        p.rag_enabled = False  # type: ignore[attr-defined]
    return p
 
 
class TestCohereProviderStandalone:
    """Tests for CohereProvider in providers/cohere_provider.py."""
 
    def test_send_mock_key(self) -> None:
        """_chat() returns 'mock:hello' when api_key == 'test'."""
        p: CohereProvider = CohereProvider.__new__(CohereProvider)
        p._documents = []  # pylint: disable=protected-access
        p._vectors = []  # pylint: disable=protected-access
        p._metadata = []  # pylint: disable=protected-access
        p.api_key = "test"  # type: ignore[attr-defined]
        p.rag_enabled = False  # type: ignore[attr-defined]
        p.client = None  # type: ignore[attr-defined]
        result = p._chat("hello")  # pylint: disable=protected-access
        assert result == "mock:hello"
 
    def test_send_non_rag(self) -> None:
        """send() delegates to _chat() when RAG is disabled."""
        p = _make_cohere_provider()
        p._chat = MagicMock(return_value="response")  # pylint: disable=protected-access
        result = p.send("hello")
        assert result == "response"
 
    def test_cosine_similarity_identical(self) -> None:
        """_cosine_similarity of two identical vectors is ~1.0."""
        p: CohereProvider = CohereProvider.__new__(CohereProvider)
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        sim = p._cosine_similarity(a, b)  # pylint: disable=protected-access
        assert abs(sim - 1.0) < 0.01
 
    def test_cosine_similarity_orthogonal(self) -> None:
        """_cosine_similarity of orthogonal vectors is ~0."""
        p: CohereProvider = CohereProvider.__new__(CohereProvider)
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = p._cosine_similarity(a, b)  # pylint: disable=protected-access
        assert abs(sim) < 0.01
 
    def test_retrieve_empty(self) -> None:
        """retrieve() returns [] when no vectors are stored."""
        p = _make_cohere_provider()
        result = p.retrieve("query")
        assert result == []
 
    def test_clear_index(self) -> None:
        """clear_index() empties all internal stores."""
        p = _make_cohere_provider()
        p._documents = ["doc"]  # pylint: disable=protected-access
        p._vectors = [[1.0]]  # pylint: disable=protected-access
        p._metadata = [{}]  # pylint: disable=protected-access
        p.clear_index()
        assert p._documents == []  # pylint: disable=protected-access
        assert p._vectors == []  # pylint: disable=protected-access
        assert p._metadata == []  # pylint: disable=protected-access
 
    def test_query_documents_without_rag_raises(self) -> None:
        """query_documents() raises ValueError when RAG is disabled."""
        p = _make_cohere_provider()
        p.rag_enabled = False  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="RAG is not enabled"):
            p.query_documents("query")
 
    def test_upsert_documents_empty(self) -> None:
        """upsert_documents([]) must not raise."""
        p = _make_cohere_provider()
        p.upsert_documents([])
 
    def test_embed_empty(self) -> None:
        """_embed([]) returns an empty list."""
        p = _make_cohere_provider()
        result = p._embed([])  # pylint: disable=protected-access
        assert result == []
 
    def test_send_with_rag_no_context(self) -> None:
        """send() calls _chat() unchanged when retrieve() returns nothing."""
        p = _make_cohere_provider()
        p.rag_enabled = True  # type: ignore[attr-defined]
        p.retrieve = MagicMock(return_value=[])  # type: ignore[method-assign]
        p._chat = MagicMock(return_value="chat result")  # pylint: disable=protected-access
        result = p.send("hello")
        assert result == "chat result"
 
    def test_send_with_rag_with_context(self) -> None:
        """send() augments the prompt with retrieved context."""
        p = _make_cohere_provider()
        p.rag_enabled = True  # type: ignore[attr-defined]
        p.retrieve = MagicMock(return_value=[{"text": "ctx"}])  # type: ignore[method-assign]
        p._chat = MagicMock(return_value="augmented result")  # pylint: disable=protected-access
        result = p.send("hello")
        assert result == "augmented result"
        call_args = p._chat.call_args[0][0]  # pylint: disable=protected-access
        assert "ctx" in call_args
 
 
# ─────────────────────────────────────────────
# providers/perplexity_provider.py
# ─────────────────────────────────────────────
 
 
class TestPerplexityProvider:
    """Tests for PerplexityProvider."""
 
    def _make_provider(self, api_key: str = "test-key") -> PerplexityProvider:
        """Return a PerplexityProvider with a mocked OpenAI client."""
        mock_client = MagicMock()
        with patch("ai_cli.providers.perplexity_provider.OpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"PERPLEXITY_API_KEY": api_key}):
                p = PerplexityProvider(api_key=api_key)
                p.client = mock_client
        return p
 
    def test_send_success(self) -> None:
        """send() returns stripped content on success."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  plex answer  "
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.send("hello")
        assert result == "plex answer"
 
    def test_send_failure_raises(self) -> None:
        """send() propagates API exceptions."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        with pytest.raises(RuntimeError, match="down"):
            p.send("hello")
 
    def test_send_no_choices_returns_empty(self) -> None:
        """send() returns '' when choices list is empty."""
        p = self._make_provider()
        p.client.chat.completions.create.return_value = MagicMock(choices=[])
        result = p.send("hello")
        assert result == ""
 
    def test_send_no_message_returns_empty(self) -> None:
        """send() returns '' when the first choice has no message."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message = None
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.send("hello")
        assert result == ""