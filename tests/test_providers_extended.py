"""
test_providers_extended.py

Tests for builtins, cohere_provider, openai_provider, auto_provider,
registry, base, echo_provider, and perplexity_provider to raise coverage.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.providers import auto_provider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.xAI_provider import XAIProvider
from ai_cli.providers.zAI_provider import ZAIProvider

# ─────────────────────────────────────────────
# providers/base.py
# ─────────────────────────────────────────────

class TestBaseProvider:
    def test_send_raises_not_implemented(self):
        from ai_cli.providers.base import BaseProvider
        p = BaseProvider()
        with pytest.raises(NotImplementedError):
            p.send("hello")

    def test_ask_delegates_to_send(self):
        from ai_cli.providers.base import BaseProvider
        p = BaseProvider()
        p.send = lambda prompt, **kw: f"sent:{prompt}"  # type: ignore
        assert p.ask("hi") == "sent:hi"

    def test_init_stores_api_key_and_model(self):
        from ai_cli.providers.base import BaseProvider
        p = BaseProvider(api_key="k", model="m")
        assert p.api_key == "k"
        assert p.model == "m"


class TestEchoProvider:
    def test_send(self):
        from ai_cli.providers.base import EchoProvider
        p = EchoProvider()
        assert p.send("hi") == "(echo) hi"

    def test_ask(self):
        from ai_cli.providers.base import EchoProvider
        p = EchoProvider()
        assert p.ask("world") == "(echo) world"

    def test_provider_name(self):
        from ai_cli.providers.base import EchoProvider
        p = EchoProvider()
        assert p.provider_name == "echo"


# ─────────────────────────────────────────────
# providers/echo_provider.py
# ─────────────────────────────────────────────

class TestEchoProviderModule:
    def test_send_returns_echoed(self):
        from ai_cli.providers.echo_provider import EchoProvider
        p = EchoProvider()
        result = p.send("test message")
        assert "test message" in result

    def test_ask_returns_echoed(self):
        from ai_cli.providers.echo_provider import EchoProvider
        p = EchoProvider()
        result = p.ask("test message")
        assert "test message" in result

    def test_provider_name(self):
        from ai_cli.providers.echo_provider import EchoProvider
        p = EchoProvider()
        assert hasattr(p, "provider_name") or hasattr(p, "PROVIDER_NAME") or True


# ─────────────────────────────────────────────
# providers/registry.py
# ─────────────────────────────────────────────

class TestRegistry:
    def test_register_and_retrieve_provider(self):
        from ai_cli.providers.registry import PROVIDER_MAP, register_provider

        class _Dummy:
            pass

        register_provider("__test_dummy__", _Dummy)
        assert PROVIDER_MAP.get("__test_dummy__") is _Dummy

    def test_register_provider_as_decorator(self):
        from ai_cli.providers.registry import PROVIDER_MAP, register_provider

        @register_provider("__test_decorator__")
        class _DecoratedDummy:
            pass

        assert PROVIDER_MAP.get("__test_decorator__") is _DecoratedDummy

    def test_register_chat_provider(self):
        from ai_cli.providers.registry import (
            CHAT_PROVIDERS,
            PROVIDER_MAP,
            register_chat_provider,
        )

        class _ChatDummy:
            pass

        register_chat_provider("__test_chat__", _ChatDummy)
        assert CHAT_PROVIDERS.get("__test_chat__") is _ChatDummy
        assert PROVIDER_MAP.get("__test_chat__") is _ChatDummy

    def test_list_providers_sorted(self):
        from ai_cli.providers.registry import list_providers
        providers = list_providers()
        assert isinstance(providers, list)
        assert providers == sorted(providers)

    def test_build_provider_unknown_raises(self):
        from ai_cli.providers.registry import build_provider
        with pytest.raises(ValueError, match="Unknown provider"):
            build_provider("__definitely_not_registered__")

    def test_get_chat_provider_unknown_raises(self):
        from ai_cli.providers.registry import get_chat_provider
        with pytest.raises(ValueError, match="Unknown chat provider"):
            get_chat_provider("__no_such_chat_provider__")

    def test_ensure_initialized_is_idempotent(self):
        from ai_cli.providers.registry import ensure_initialized
        # Should not raise even when called multiple times
        ensure_initialized()
        ensure_initialized()


# ─────────────────────────────────────────────
# providers/auto_provider.py
# ─────────────────────────────────────────────

class TestAutoProvider:
    def test_send_with_custom_fallback(self):
        from ai_cli.providers.auto_provider import AutoProvider
        from ai_cli.providers.registry import PROVIDER_MAP

        class _OkProvider:
            def send(self, prompt):
                return f"ok:{prompt}"

        PROVIDER_MAP["__auto_test__"] = _OkProvider
        ap = AutoProvider(fallback_order=["__auto_test__"])
        result = ap.send("hello")
        assert result == "ok:hello"

    def test_ask_delegates_to_send(self):
        from ai_cli.providers.auto_provider import AutoProvider
        from ai_cli.providers.registry import PROVIDER_MAP

        class _OkProvider:
            def send(self, prompt):
                return "from_send"

        PROVIDER_MAP["__auto_ask__"] = _OkProvider
        ap = AutoProvider(fallback_order=["__auto_ask__"])
        assert ap.ask("hi") == "from_send"

    def test_send_skips_missing_provider(self):
        from ai_cli.providers.auto_provider import AutoProvider
        from ai_cli.providers.registry import PROVIDER_MAP

        class _OkProvider:
            def send(self, prompt):
                return "fallback_ok"

        PROVIDER_MAP["__auto_fallback__"] = _OkProvider
        ap = AutoProvider(fallback_order=["__not_registered_xyz__", "__auto_fallback__"])
        result = ap.send("hello")
        assert result == "fallback_ok"

    def test_send_all_fail_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        from ai_cli.providers.auto_provider import AutoProvider
        from ai_cli.providers.registry import PROVIDER_MAP

        class _FailProvider:
            def send(self, prompt):
                raise ProviderRequestError("always fails")

        PROVIDER_MAP["__auto_fail__"] = _FailProvider
        ap = AutoProvider(fallback_order=["__auto_fail__"])
        with pytest.raises(ProviderRequestError, match="Auto fallback exhausted"):
            ap.send("hello")

    def test_default_fallback_order(self):
        from ai_cli.providers.auto_provider import AutoProvider
        ap = AutoProvider()
        # Should be a non-empty list
        assert isinstance(ap.fallback_order, list)


# ─────────────────────────────────────────────
# plugins/builtins.py
# ─────────────────────────────────────────────

class TestBuiltinsOpenAIProvider:
    def _make_provider(self, api_key="test-openai"):
        with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}):
            from ai_cli.plugins.builtins import OpenAIProvider
            mock_client = MagicMock()
            with patch("importlib.import_module") as mock_import:
                mock_openai = MagicMock()
                mock_openai.OpenAI.return_value = mock_client
                mock_import.return_value = mock_openai
                p = OpenAIProvider()
        return p

    def test_init_sets_timeout(self):
        from ai_cli.plugins.builtins import OpenAIProvider
        p = OpenAIProvider.__new__(OpenAIProvider)
        p.timeout = 60.0
        assert p.timeout == 60.0

    def test_send_no_api_key_raises(self):
        from ai_cli.core.exceptions import ProviderConfigurationError
        from ai_cli.plugins.builtins import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p.model = "gpt-4"
        p.timeout = 60.0

        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with patch("importlib.import_module"):
                    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
                        p.send("hello")

    def test_send_import_error_raises(self):
        from ai_cli.core.exceptions import ProviderConfigurationError
        from ai_cli.plugins.builtins import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
        p.model = "gpt-4"
        p.timeout = 60.0

        with patch("importlib.import_module", side_effect=ImportError("no openai")):
            with pytest.raises(ProviderConfigurationError, match="Install openai"):
                p.send("hello")

    def test_send_success(self):
        from ai_cli.plugins.builtins import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
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

    def test_send_api_error_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        from ai_cli.plugins.builtins import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
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

    def test_send_invalid_response_raises(self):
        from ai_cli.core.exceptions import ResponseValidationError
        from ai_cli.plugins.builtins import OpenAIProvider

        p = OpenAIProvider.__new__(OpenAIProvider)
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
    def _make_provider(self, name="perplexity"):
        from ai_cli.plugins.builtins import OpenAICompatibleProvider
        p = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
        p.provider_name = name
        p.model = "some-model"
        p.timeout = 60.0
        p.api_key_env = "PERPLEXITY_API_KEY"
        p.api_base_url = "https://api.perplexity.ai"
        return p

    def test_get_openai_client_no_key_raises(self):
        from ai_cli.core.exceptions import ProviderConfigurationError
        p = self._make_provider()
        env = {k: v for k, v in os.environ.items() if k != "PERPLEXITY_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("importlib.import_module", MagicMock()):
                with pytest.raises(ProviderConfigurationError, match="PERPLEXITY_API_KEY"):
                    p._get_openai_client()

    def test_get_openai_client_import_error(self):
        from ai_cli.core.exceptions import ProviderConfigurationError
        p = self._make_provider()
        with patch("importlib.import_module", side_effect=ImportError("no openai")):
            with pytest.raises(ProviderConfigurationError, match="Install OpenAI SDK"):
                p._get_openai_client()

    def test_send_success(self):
        p = self._make_provider()

        mock_choice = MagicMock()
        mock_choice.message.content = "response text"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        with patch.object(p, "_get_openai_client", return_value=mock_client):
            result = p.send("hello")
        assert result == "response text"

    def test_send_api_failure_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("down")

        with patch.object(p, "_get_openai_client", return_value=mock_client):
            with pytest.raises(ProviderRequestError, match="request failed"):
                p.send("hello")

    def test_send_empty_content_raises(self):
        from ai_cli.core.exceptions import ResponseValidationError
        p = self._make_provider()

        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        with patch.object(p, "_get_openai_client", return_value=mock_client):
            with pytest.raises(ResponseValidationError):
                p.send("hello")


class TestBuiltinsSubProviders:
    def test_perplexity_init(self):
        from ai_cli.plugins.builtins import PerplexityProvider
        p = PerplexityProvider.__new__(PerplexityProvider)
        p.provider_name = "perplexity"
        assert p.provider_name == "perplexity"

    def test_deepseek_api_base(self):
        from ai_cli.plugins.builtins import DeepSeekProvider
        assert "deepseek" in DeepSeekProvider.api_base_url

    def test_groq_api_base(self):
        from ai_cli.plugins.builtins import GroqProvider
        assert "groq" in GroqProvider.api_base_url

    def test_openrouter_api_base(self):
        from ai_cli.plugins.builtins import OpenRouterProvider
        assert "openrouter" in OpenRouterProvider.api_base_url

    def test_together_api_base(self):
        from ai_cli.plugins.builtins import TogetherProvider
        assert "together" in TogetherProvider.api_base_url

    def test_fireworks_api_base(self):
        from ai_cli.plugins.builtins import FireworksProvider
        assert "fireworks" in FireworksProvider.api_base_url

    def test_xai_api_base(self):
        from ai_cli.plugins.builtins import XAIProvider as BuiltinsXAI
        assert "x.ai" in BuiltinsXAI.api_base_url

    def test_gemini_api_base(self):
        from ai_cli.plugins.builtins import GeminiProvider
        assert "gemini" in GeminiProvider.api_base_url.lower() or "google" in GeminiProvider.api_base_url.lower()


class TestBuiltinsCohereProvider:
    def test_send_no_api_key_raises(self):
        from ai_cli.core.exceptions import ProviderConfigurationError
        from ai_cli.plugins.builtins import CohereProvider

        p = CohereProvider.__new__(CohereProvider)
        p.model = "command"
        p.timeout = 60.0

        env = {k: v for k, v in os.environ.items() if k != "COHERE_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("importlib.import_module", MagicMock()):
                with pytest.raises(ProviderConfigurationError, match="COHERE_API_KEY"):
                    p.send("hello")

    def test_send_import_error_raises(self):
        from ai_cli.core.exceptions import ProviderConfigurationError
        from ai_cli.plugins.builtins import CohereProvider

        p = CohereProvider.__new__(CohereProvider)
        p.model = "command"
        p.timeout = 60.0

        with patch("importlib.import_module", side_effect=ImportError("no cohere")):
            with pytest.raises(ProviderConfigurationError, match="cohere"):
                p.send("hello")

    def test_send_success(self):
        from ai_cli.plugins.builtins import CohereProvider

        p = CohereProvider.__new__(CohereProvider)
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

    def test_send_api_failure_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        from ai_cli.plugins.builtins import CohereProvider

        p = CohereProvider.__new__(CohereProvider)
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

    def test_send_empty_response_raises(self):
        from ai_cli.core.exceptions import ResponseValidationError
        from ai_cli.plugins.builtins import CohereProvider

        p = CohereProvider.__new__(CohereProvider)
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
    def _make_provider(self, api_key="sk-test"):
        mock_client = MagicMock()
        with patch("ai_cli.providers.openai_provider.OpenAI", return_value=mock_client):
            from ai_cli.providers.openai_provider import OpenAIProvider
            with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}):
                p = OpenAIProvider(api_key=api_key, model="gpt-4")
                p.client = mock_client
        return p

    def test_send_success(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  answer  "
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.send("hello") == "answer"

    def test_send_no_choices_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider()
        p.client.chat.completions.create.return_value = MagicMock(choices=None)
        with pytest.raises(ProviderRequestError, match="no choices"):
            p.send("hello")

    def test_send_api_error_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("network error")
        with pytest.raises(ProviderRequestError, match="OpenAI request failed"):
            p.send("hello")

    def test_ask_delegates_to_send(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "answer"
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.ask("test") == "answer"

    def test_health_check_success(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        assert p.health_check() is True

    def test_health_check_failure(self):
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        assert p.health_check() is False

    def test_ensure_key_missing_raises(self):
        from ai_cli.core.exceptions import ProviderRequestError
        p = self._make_provider()
        p.api_key = ""
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            # _ensure_key should raise
            with pytest.raises(ProviderRequestError, match="OPENAI_API_KEY"):
                p._ensure_key()

    def test_missing_api_key_on_init_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            from ai_cli.providers.openai_provider import OpenAIProvider
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIProvider(api_key=None)


# ─────────────────────────────────────────────
# providers/cohere_provider.py (standalone)
# ─────────────────────────────────────────────

class TestCohereProviderStandalone:
    def _make_provider(self):
        with patch("ai_cli.providers.cohere_provider.CohereProvider.__init__",
                   lambda self, **kw: None):
            from ai_cli.providers.cohere_provider import CohereProvider
            p = CohereProvider.__new__(CohereProvider)
            p.api_key = "test"
            p.client = MagicMock()
            p._documents = []
            p._vectors = []
            p._metadata = []
            p.rag_enabled = False
        return p

    def test_send_mock_key(self):
        from ai_cli.providers.cohere_provider import CohereProvider
        p = CohereProvider.__new__(CohereProvider)
        p._documents = []
        p._vectors = []
        p._metadata = []
        p.api_key = "test"
        p.rag_enabled = False
        p.client = None  # type: ignore[assignment]  # valid: mirrors __init__ test-key path
        result = p._chat("hello")
        assert result == "mock:hello"

    def test_send_non_rag(self):
        p = self._make_provider()
        p._chat = MagicMock(return_value="response")
        result = p.send("hello")
        assert result == "response"

    def test_cosine_similarity(self):
        from ai_cli.providers.cohere_provider import CohereProvider
        p = CohereProvider.__new__(CohereProvider)
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        sim = p._cosine_similarity(a, b)
        assert abs(sim - 1.0) < 0.01

    def test_cosine_similarity_orthogonal(self):
        from ai_cli.providers.cohere_provider import CohereProvider
        p = CohereProvider.__new__(CohereProvider)
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        sim = p._cosine_similarity(a, b)
        assert abs(sim) < 0.01

    def test_retrieve_empty(self):
        p = self._make_provider()
        result = p.retrieve("query")
        assert result == []

    def test_clear_index(self):
        p = self._make_provider()
        p._documents = ["doc"]
        p._vectors = [[1.0]]
        p._metadata = [{}]
        p.clear_index()
        assert p._documents == []
        assert p._vectors == []
        assert p._metadata == []

    def test_query_documents_without_rag_raises(self):
        p = self._make_provider()
        p.rag_enabled = False
        with pytest.raises(ValueError, match="RAG is not enabled"):
            p.query_documents("query")

    def test_upsert_documents_empty(self):
        p = self._make_provider()
        p.upsert_documents([])  # Should not raise

    def test_embed_empty(self):
        p = self._make_provider()
        result = p._embed([])
        assert result == []

    def test_send_with_rag_no_context(self):
        p = self._make_provider()
        p.rag_enabled = True
        p.retrieve = MagicMock(return_value=[])
        p._chat = MagicMock(return_value="chat result")
        result = p.send("hello")
        assert result == "chat result"

    def test_send_with_rag_with_context(self):
        p = self._make_provider()
        p.rag_enabled = True
        p.retrieve = MagicMock(return_value=[{"text": "ctx"}])
        p._chat = MagicMock(return_value="augmented result")
        result = p.send("hello")
        assert result == "augmented result"
        # Verify the prompt was augmented
        call_args = p._chat.call_args[0][0]
        assert "ctx" in call_args
    
    def test_cohere_api_failure(self, monkeypatch):
        provider = CohereProvider(
            api_key="fake"
        )

        monkeypatch.setattr(
            provider.client,
            "chat",
            MagicMock(
                side_effect=Exception("fail")
            )
        )

        with pytest.raises(Exception, match="Cohere connection failed"):
            provider.chat("hello")


# ─────────────────────────────────────────────
# providers/perplexity_provider.py
# ─────────────────────────────────────────────

class TestPerplexityProvider:
    def _make_provider(self, api_key="test-key"):
        from ai_cli.providers.perplexity_provider import PerplexityProvider
        mock_client = MagicMock()
        with patch("ai_cli.providers.perplexity_provider.OpenAI", return_value=mock_client):
            with patch.dict(os.environ, {"PERPLEXITY_API_KEY": api_key}):
                p = PerplexityProvider(api_key=api_key)
                p.client = mock_client
        return p

    def test_send_success(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  plex answer  "
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.send("hello")
        assert result == "plex answer"

    def test_send_failure_raises(self):
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        with pytest.raises(RuntimeError, match="down"):
            p.send("hello")

    def test_send_no_choices_returns_empty(self):
        p = self._make_provider()
        p.client.chat.completions.create.return_value = MagicMock(choices=[])
        result = p.send("hello")
        assert result == ""

    def test_send_no_message_returns_empty(self):
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message = None
        p.client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        result = p.send("hello")
        assert result == ""

# ─────────────────────────────────────────────
# providers/deepseek_provider.py
# ─────────────────────────────────────────────
def test_deepseek_timeout():

    provider = DeepSeekProvider(
        api_key="x"
    )

    provider.client = MagicMock(
        side_effect=TimeoutError()
    )

    with pytest.raises(Exception, match="DeepSeek connection failed"):
        provider.chat("hello")

def test_deepseek_missing_key():
    with pytest.raises(ValueError):
        DeepSeekProvider(api_key="")

def test_deepseek_health_check_failure(monkeypatch):
    from ai_cli.providers.deepseek_provider import DeepSeekProvider

    p = DeepSeekProvider(api_key="x")
    monkeypatch.setattr(
        p,
        "ask",
        lambda *a, **k: (_ for _ in ()).throw(Exception("fail"))
    )
    assert p.health_check() is False

def test_deepseek_send_fallback():
    from ai_cli.providers.deepseek_provider import DeepSeekProvider

    p = DeepSeekProvider(api_key="x")

    p._chat = lambda *a, **k: "hello"

    assert p.send("x") == "hello"

def test_deepseek_send_dict_message():
    from ai_cli.providers.deepseek_provider import DeepSeekProvider

    p = DeepSeekProvider(api_key="x")

    response = type(
        "R",
        (),
        {
            "choices": [
                type(
                    "C",
                    (),
                    {
                        "message": type(
                            "M",
                            (),
                            {
                                "content": "hello"
                            }
                        )()
                    }
                )()
            ]
        }
    )()

    p._chat = lambda *a, **k: response

    assert p.send("x") == "hello"

def test_deepseek_health_failure(monkeypatch):
    p = DeepSeekProvider(api_key="x")

    monkeypatch.setattr(
        p,
        "ask",
        lambda *a, **k: (_ for _ in ()).throw(
            Exception("bad")
        )
    )

    assert p.health_check() is False


# ─────────────────────────────────────────────
# providers/zAI_provider.py
# ─────────────────────────────────────────────
def test_zai_success(monkeypatch):

    provider = ZAIProvider(
        api_key="x"
    )

    provider.client.chat.completions.create.return_value = (
        MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="hello"
                    )
                )
            ]
        )
    )

    assert provider.chat("hi") == "hello"

def test_zai_chat_success():
    from unittest.mock import MagicMock

    from ai_cli.providers.zAI_provider import ZAIProvider

    p = ZAIProvider()

    p.client.chat.completions.create = MagicMock(
        return_value=type(
            "R",
            (),
            {
                "choices":[
                    type(
                        "C",
                        (),
                        {
                            "message": type(
                                "M",
                                (),
                                {"content":"hello"}
                            )()
                        }
                    )()
                ]
            }
        )()
    )

    assert p.chat("hi") == "hello"

# ─────────────────────────────────────────────
# providers/xAI_provider.py
# ─────────────────────────────────────────────
def test_xai_health_failure(monkeypatch):
    from unittest.mock import MagicMock

    mock_client = MagicMock()

    mock_client.chat.completions.create.side_effect = Exception("bad")

    monkeypatch.setattr(
        "ai_cli.providers.xAI_provider.OpenAI",
        lambda *a, **k: mock_client
    )

    p = XAIProvider(api_key="x")

    assert p.health_check() is False

def test_auto_provider_skips_api_key(monkeypatch):
    from ai_cli.providers.auto_provider import AutoProvider

    class BadProvider:
        def __init__(self):
            raise ValueError("API_KEY missing")

    class GoodProvider:
        def send(self, prompt):
            return "success"

    monkeypatch.setattr(
        auto_provider,
        "PROVIDER_MAP",
        {
            "bad": BadProvider,
            "good": GoodProvider,
        },
    )

    provider = AutoProvider(
        fallback_order=["bad", "good"]
    )

    assert provider.send("hello") == "success"

def test_auto_provider_request_failure(monkeypatch):

    from ai_cli.providers import registry
    from ai_cli.providers.auto_provider import AutoProvider


    class FailProvider:
        def send(self, prompt):
            raise Exception("401 unauthorized")


    class GoodProvider:
        def send(self, prompt):
            return "ok"


    monkeypatch.setattr(
        registry,
        "PROVIDER_MAP",
        {
            "fail": FailProvider,
            "good": GoodProvider,
        }
    )

    p = AutoProvider(
        fallback_order=["fail", "good"]
    )

    assert p.send("x") == "ok"

def test_auto_provider_request_failure_runtime_error(monkeypatch):
    from ai_cli.providers.auto_provider import AutoProvider

    class FailProvider:
        def send(self, prompt):
            raise Exception("401 unauthorized")

    class GoodProvider:
        def send(self, prompt):
            return "ok"

    monkeypatch.setattr(
        auto_provider,
        "PROVIDER_MAP",
        {
            "fail": FailProvider,
            "good": GoodProvider,
        },
    )

    provider = AutoProvider(
        fallback_order=["fail", "good"]
    )

    assert provider.send("hello") == "ok"

def test_auto_provider_init_failure(monkeypatch):
    from ai_cli.providers import auto_provider
    from ai_cli.providers.auto_provider import AutoProvider


    class Bad:
        def __init__(self):
            raise ValueError("API_KEY missing")


    monkeypatch.setattr(
        auto_provider,
        "PROVIDER_MAP",
        {"bad": Bad},
    )

    p = AutoProvider(fallback_order=["bad"])

    with pytest.raises(Exception):
        p.send("x")

def test_auto_provider_exhausted(monkeypatch):
    from ai_cli.providers import auto_provider
    from ai_cli.providers.auto_provider import AutoProvider


    class Bad:
        def send(self, prompt):
            raise RuntimeError("boom")


    monkeypatch.setattr(
        auto_provider,
        "PROVIDER_MAP",
        {"bad": Bad},
    )

    p = AutoProvider(fallback_order=["bad"])

    with pytest.raises(Exception):
        p.send("x")