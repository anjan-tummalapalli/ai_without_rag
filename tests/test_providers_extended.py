"""
test_providers_extended.py

Tests for cohere_provider, openai_provider, auto_provider,
registry, base, echo_provider, and perplexity_provider to raise coverage.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ai_cli.core.exceptions import (
    ProviderRequestError,
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

# --------------------------------------------
# providers/base.py
# --------------------------------------------


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
        p.send = (  # type: ignore[method-assign]
            lambda prompt, **kw: f"sent:{prompt}"
        )
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


# --------------------------------------------
# providers/echo_provider.py
# --------------------------------------------


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


# --------------------------------------------
# providers/registry.py
# --------------------------------------------


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
        """register_chat_provider populates CHAT_PROVIDERS and PROVIDER_MAP."""
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


# --------------------------------------------
# providers/auto_provider.py
# --------------------------------------------


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
        ap = AutoProvider(
            fallback_order=["__not_registered_xyz__", "__auto_fallback__"]
        )
        result = ap.send("hello")
        assert result == "fallback_ok"

    def test_send_all_fail_raises(self) -> None:
        """send() raises ProviderRequestError when all fallbacks fail."""
        PROVIDER_MAP["__auto_fail__"] = _AlwaysFailProvider
        ap = AutoProvider(fallback_order=["__auto_fail__"])
        with pytest.raises(
            ProviderRequestError, match="Auto fallback exhausted"
        ):
            ap.send("hello")

    def test_default_fallback_order(self) -> None:
        """AutoProvider initialises fallback_order to a list."""
        ap = AutoProvider()
        assert isinstance(ap.fallback_order, list)



# --------------------------------------------
# providers/openai_provider.py
# --------------------------------------------


class TestOpenAIProviderModule:
    """Tests for OpenAIProvider in providers/openai_provider.py."""

    def _make_provider(self, api_key: str = "sk-test") -> OpenAIProvider:
        """Return an OpenAIProvider with a mocked OpenAI client."""
        mock_client = MagicMock()
        with patch(
            "ai_cli.providers.openai_provider.OpenAI",
            return_value=mock_client,
        ):
            with patch.dict(os.environ, {"OPENAI_API_KEY": api_key}):
                p = OpenAIProvider(api_key=api_key, model="gpt-4")
                p.client = mock_client
        return p

    def test_send_success(self) -> None:
        """send() returns stripped content on success."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  answer  "
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
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
        p.client.chat.completions.create.side_effect = RuntimeError(
            "network error"
        )
        with pytest.raises(ProviderRequestError, match="OpenAI request failed"):
            p.send("hello")

    def test_ask_delegates_to_send(self) -> None:
        """ask() is an alias for send()."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "answer"
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        assert p.ask("test") == "answer"

    def test_health_check_success(self) -> None:
        """health_check() returns True when the API responds with choices."""
        p = self._make_provider()
        mock_choice = MagicMock()
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        assert p.health_check() is True

    def test_health_check_failure(self) -> None:
        """health_check() returns False when the API raises."""
        p = self._make_provider()
        p.client.chat.completions.create.side_effect = RuntimeError("down")
        assert p.health_check() is False

    def test_ensure_key_missing_raises(self) -> None:
        """_ensure_key raises ProviderRequestError when API key absent."""
        p = self._make_provider()
        p.api_key = ""
        env: dict[str, str] = {
            k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ProviderRequestError, match="OPENAI_API_KEY"):
                p._ensure_key()  # pylint: disable=protected-access

    def test_missing_api_key_on_init_raises(self) -> None:
        """OpenAIProvider.__init__ raises ValueError when no API key found."""
        env: dict[str, str] = {
            k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIProvider(api_key=None)


# --------------------------------------------
# providers/cohere_provider.py (standalone)
# --------------------------------------------


def _make_cohere_provider() -> CohereProvider:
    """Return a CohereProvider in test-key mode with internal stores reset."""
    with patch(
        "ai_cli.providers.cohere_provider.CohereProvider.__init__",
        lambda self, **kw: None,
    ):
        p: CohereProvider = CohereProvider.__new__(CohereProvider)
        p.api_key = "test"  # type: ignore[attr-defined]
        p.client = MagicMock()  # type: ignore[attr-defined]
        # pylint: disable=protected-access
        p._documents = []  # type: ignore[attr-defined]
        p._vectors = []  # type: ignore[attr-defined]
        p._metadata = []  # type: ignore[attr-defined]
        # pylint: enable=protected-access
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
        p._chat = MagicMock(  # pylint: disable=protected-access
            return_value="response"
        )
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
        p._chat = MagicMock(  # pylint: disable=protected-access
            return_value="chat result"
        )
        result = p.send("hello")
        assert result == "chat result"

    def test_send_with_rag_with_context(self) -> None:
        """send() augments the prompt with retrieved context."""
        p = _make_cohere_provider()
        p.rag_enabled = True  # type: ignore[attr-defined]
        p.retrieve = MagicMock(  # type: ignore[method-assign]
            return_value=[{"text": "ctx"}]
        )
        p._chat = MagicMock(  # pylint: disable=protected-access
            return_value="augmented result"
        )
        result = p.send("hello")
        assert result == "augmented result"
        # pylint: disable=protected-access
        call_args = p._chat.call_args[0][0]
        assert "ctx" in call_args

    def test_cohere_requires_api_key(self, monkeypatch):
        """Cover missing COHERE_API_KEY validation."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            CohereProvider(api_key=None)

    def test_cohere_import_failure(self, monkeypatch):
        """Import failure should raise RuntimeError."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cohere":
                raise ImportError("missing")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError):
            CohereProvider(api_key="real-key")

    def test_cohere_chat_wraps_exception(self):
        provider = CohereProvider(api_key="test")
        provider.send = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="Cohere connection failed"):
            provider.chat("hello")

    def test_cohere_chat_client_response(self):
        provider = CohereProvider(api_key="test")
        provider.client = MagicMock()
        provider.client.chat.return_value.text = "success"
        assert provider._chat("hello") == "success"
        provider.client.chat.assert_called_once()

    def test_embed_empty_returns_empty(self):
        provider = CohereProvider(api_key="test")
        assert provider._embed([]) == []

    def test_embed_success(self):
        provider = CohereProvider(api_key="test")
        provider.client = MagicMock()
        provider.client.embed.return_value.embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        result = provider._embed(["a", "b"])
        assert len(result) == 2
        provider.client.embed.assert_called_once()

    def test_send_without_rag_calls_chat(self):
        provider = CohereProvider(
            api_key="test",
            rag_enabled=False,
        )
        provider._chat = MagicMock(return_value="answer")
        assert provider.send("hello") == "answer"
        provider._chat.assert_called_once_with("hello")

    def test_send_with_rag_context(self):
        provider = CohereProvider(
            api_key="test",
            rag_enabled=True,
        )

        provider.retrieve = MagicMock(
            return_value=[
                {"text": "document one"},
                {"text": "document two"},
            ]
        )

        provider._chat = MagicMock(return_value="rag-answer")
        result = provider.send("What is AI?")
        assert result == "rag-answer"
        prompt = provider._chat.call_args.args[0]
        assert "document one" in prompt
        assert "document two" in prompt
        assert "Question:" in prompt

    def test_chat_returns_mock_when_client_none(self):
        provider = CohereProvider(api_key="test")
        provider.client = None
        assert provider._chat("hello") == "mock:hello"

    def test_cohere_upsert_documents_without_metadata(self):
        p = CohereProvider(
            api_key="test",
            rag_enabled=True,
        )

        p._chunk_text = lambda text: [text + "_1", text + "_2"]
        p._embed = lambda texts: [[1.0] for _ in texts]

        p.upsert_documents(
            ["doc1", "doc2"],
            metadatas=None,
        )

        assert len(p._documents) == 4
        assert len(p._vectors) == 4
        assert len(p._metadata) == 4

        assert p._metadata[0]["doc_index"] == 0
        assert p._metadata[2]["doc_index"] == 1

    def test_cohere_upsert_embedding_mismatch(self):
        p = CohereProvider(
            api_key="test",
            rag_enabled=True,
        )

        p._chunk_text = lambda text: ["a", "b"]
        p._embed = lambda texts: [[1.0]]

        with pytest.raises(RuntimeError):
            p.upsert_documents(["hello"])

    def test_cohere_retrieve_scores_sorted(self):
        p = CohereProvider(
            api_key="test",
            rag_enabled=True,
        )

        p._documents = ["one", "two"]
        p._metadata = [{}, {}]
        p._vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        p._embed = lambda texts: [[1.0, 0.0]]

        results = p.retrieve(
            "query",
            top_k=2,
        )

        assert results[0]["text"] == "one"
        assert len(results) == 2

    def test_query_documents_rag_enabled(self):
        p = CohereProvider(
            api_key="test",
            rag_enabled=True,
        )
        expected = [{"text": "hello"}]
        p.retrieve = lambda query, top_k=5: expected
        assert p.query_documents("abc") == expected

    def test_cohere_import_failure_test_key(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "cohere":
                raise ImportError()
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(
            builtins,
            "__import__",
            fake_import,
        )

        p = CohereProvider(api_key="test")

        assert p.client is None


# --------------------------------------------
# providers/perplexity_provider.py
# --------------------------------------------


class TestPerplexityProvider:
    """Tests for PerplexityProvider."""

    def _make_provider(self, api_key: str = "test-key") -> PerplexityProvider:
        """Return a PerplexityProvider with a mocked OpenAI client."""
        mock_client = MagicMock()
        with patch(
            "ai_cli.providers.perplexity_provider.OpenAI",
            return_value=mock_client,
        ):
            with patch.dict(os.environ, {"PERPLEXITY_API_KEY": api_key}):
                p = PerplexityProvider(api_key=api_key)
                p.client = mock_client
        return p

    def test_send_success(self) -> None:
        """send() returns stripped content on success."""
        p = self._make_provider()
        mock_choice = MagicMock()
        mock_choice.message.content = "  plex answer  "
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
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
        p.client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        result = p.send("hello")
        assert result == ""
