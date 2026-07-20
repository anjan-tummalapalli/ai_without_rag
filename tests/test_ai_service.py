"""Tests for ai_cli.core.service.ai_service.AIService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import ai_cli.core.service.ai_service as ai_service
from ai_cli.core.service.ai_service import AIService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_svc(**kwargs) -> AIService:
    """Return an AIService configured for the echo provider by default."""
    kwargs.setdefault("provider", "echo")
    kwargs.setdefault("timeout", 5)
    return AIService(**kwargs)


def _patch_ask(return_value):
    """Patch ``core.api.ask`` and return the patcher context manager."""
    return patch(
        "ai_cli.core.service.ai_service._core_ask",
        return_value=return_value,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAIServiceInit:
    def test_defaults(self):
        svc = AIService()
        assert svc.provider == "auto"
        assert svc.model is None
        assert svc.timeout == 60
        assert svc.stream is False
        assert svc.max_retries == 3

    def test_strips_whitespace_from_provider(self):
        svc = AIService(provider="  openai  ")
        assert svc.provider == "openai"

    def test_strips_whitespace_from_model(self):
        svc = AIService(model="  gpt-4o  ")
        assert svc.model == "gpt-4o"

    def test_empty_provider_defaults_to_auto(self):
        svc = AIService(provider="   ")
        assert svc.provider == "auto"

    def test_invalid_max_retries_raises(self):
        with pytest.raises(ValueError, match="max_retries"):
            AIService(max_retries=0)

    def test_modules_parsed_from_string(self):
        svc = AIService(modules="mod1, mod2, mod3")
        assert svc.modules == ["mod1", "mod2", "mod3"]

    def test_modules_none_when_blank(self):
        svc = AIService(modules="   ")
        assert svc.modules is None

    def test_repr(self):
        svc = AIService(provider="openai", model="gpt-4o", timeout=30)
        r = repr(svc)
        assert "openai" in r
        assert "gpt-4o" in r


# ---------------------------------------------------------------------------
# ask() — synchronous full-response
# ---------------------------------------------------------------------------


class TestAsk:
    def test_returns_string_from_provider(self):
        svc = _make_svc()
        with _patch_ask("Hello from AI"):
            result = svc.ask("Hi")
        assert result == "Hello from AI"

    def test_prompt_must_be_str(self):
        svc = _make_svc()
        with pytest.raises(TypeError):
            svc.ask(42)  # type: ignore[arg-type]

    def test_bytes_response_decoded(self):
        svc = _make_svc()
        with _patch_ask(b"byte reply"):
            result = svc.ask("test")
        assert result == "byte reply"

    def test_iterable_response_joined(self):
        svc = _make_svc()
        with _patch_ask(iter(["chunk1", " ", "chunk2"])):
            result = svc.ask("test")
        assert result == "chunk1 chunk2"

    def test_dict_response_json_formatted(self):
        svc = _make_svc()
        with _patch_ask({"key": "value"}):
            result = svc.ask("test")
        assert "key" in result
        assert "value" in result

    def test_retries_on_connection_error_then_succeeds(self):
        svc = _make_svc(backoff=0.0)
        call_count = 0

        def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        with patch(
            "ai_cli.core.service.ai_service._core_ask", side_effect=flaky
        ):
            result = svc.ask("test")

        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        svc = _make_svc(max_retries=2, backoff=0.0)
        with patch(
            "ai_cli.core.service.ai_service._core_ask",
            side_effect=ConnectionError("always fails"),
        ):
            with pytest.raises(ConnectionError):
                svc.ask("test")

    def test_timeout_error_propagated_after_retries(self):
        svc = _make_svc(max_retries=2, backoff=0.0)
        with patch(
            "ai_cli.core.service.ai_service._core_ask",
            side_effect=TimeoutError("timed out"),
        ):
            with pytest.raises(TimeoutError):
                svc.ask("test")

    def test_non_transient_error_not_retried(self):
        """ValueError is not a transient error — must not be retried."""
        svc = _make_svc(max_retries=3, backoff=0.0)
        mock = MagicMock(side_effect=ValueError("bad value"))
        with patch("ai_cli.core.service.ai_service._core_ask", mock):
            with pytest.raises(ValueError):
                svc.ask("test")
        assert mock.call_count == 1


# ---------------------------------------------------------------------------
# ask_stream() — synchronous streaming
# ---------------------------------------------------------------------------


class TestAskStream:
    def test_yields_chunks_from_iterable(self):
        svc = _make_svc()
        with _patch_ask(iter(["Hello", " ", "World"])):
            chunks = list(svc.ask_stream("Hi"))
        assert "".join(chunks) == "Hello World"

    def test_yields_single_chunk_for_scalar(self):
        svc = _make_svc()
        with _patch_ask("scalar response"):
            chunks = list(svc.ask_stream("Hi"))
        assert chunks == ["scalar response"]

    def test_prompt_must_be_str(self):
        svc = _make_svc()
        with pytest.raises(TypeError):
            list(svc.ask_stream(None))  # type: ignore[arg-type]

    def test_restores_stream_flag_after_call(self):
        svc = _make_svc(stream=False)
        with _patch_ask("data"):
            list(svc.ask_stream("test"))
        assert svc.stream is False


# ---------------------------------------------------------------------------
# ask_async() — async streaming
# ---------------------------------------------------------------------------


class TestAskAsync:
    @pytest.mark.asyncio
    async def test_yields_chunks(self):
        async def _collect(svc, prompt):
            parts = []
            async for chunk in svc.ask_async(prompt):
                parts.append(chunk)
            return parts

        svc = _make_svc()
        with _patch_ask(iter(["async", " ", "chunk"])):
            parts = await _collect(svc, "hi")
        assert "".join(parts) == "async chunk"

    @pytest.mark.asyncio
    async def test_prompt_must_be_str(self):
        svc = _make_svc()
        with pytest.raises(TypeError):
            async for _ in svc.ask_async(123):  # type: ignore[arg-type]
                pass

    @pytest.mark.asyncio
    async def test_restores_stream_flag_after_call(self):
        svc = _make_svc(stream=False)
        with _patch_ask("data"):
            async for _ in svc.ask_async("test"):
                pass
        assert svc.stream is False


# ---------------------------------------------------------------------------
# CLI integration — _make_service / _call_service round-trip
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Smoke-tests that cli.main delegates through ask() correctly."""

    def test_main_uses_ai_service(self, capsys):
        """cli.main should print the provider response when given --prompt."""
        with patch(
            "ai_cli.cli.ask",
            return_value="(echo) hello",
        ):
            from ai_cli.cli import main

            code = main(["-p", "echo", "-q", "hello"])

        assert code == 0
        captured = capsys.readouterr()
        assert "(echo) hello" in captured.out

    def test_main_returns_1_on_runtime_error(self, capsys):
        with patch(
            "ai_cli.cli.ask",
            side_effect=RuntimeError("boom"),
        ):
            from ai_cli.cli import main

            code = main(["-p", "echo", "-q", "hello"])

        assert code == 1
        assert "ERROR" in capsys.readouterr().err

    def test_main_returns_124_on_timeout(self, capsys):
        with patch(
            "ai_cli.cli.ask",
            side_effect=TimeoutError("timed out"),
        ):
            from ai_cli.cli import main

            code = main(["-p", "echo", "-q", "hello"])

        assert code == 124

def test_constructor_normalizes_values():
    svc = AIService(
        provider=" ",
        model=" ",
        profile=" ",
        modules="a,b,c",
    )

    assert svc.provider == "auto"
    assert svc.model is None
    assert svc.profile is None
    assert svc.modules == ["a", "b", "c"]

async def agen():
    yield "A"
    yield "B"

@pytest.mark.asyncio
async def test_drain_async_generator():
    result = await AIService._drain_async(agen())
    assert result == "AB"

async def coro():
    return "Hello"

@pytest.mark.asyncio
async def test_drain_async_coroutine():
    result = await AIService._drain_async(coro())
    assert result == "Hello"

@pytest.mark.asyncio
async def test_drain_async_iterable():
    result = await AIService._drain_async(["A","B"])
    assert result == "AB"

def test_drain_sync_list():
    out = AIService._drain_sync([1, 2, 3])
    assert out == "123"

def test_stream_handles_async_generator_response(monkeypatch):
    svc = AIService()
    monkeypatch.setattr(
        svc,
        "_call_with_retries",
        lambda _: agen(),
    )
    assert list(svc.ask_stream("x")) == ["AB"]

def test_build_kwargs_signature_failure(monkeypatch) -> None:
    svc = AIService(provider="openai")

    monkeypatch.setattr(
        ai_service.inspect,
        "signature",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError()),
    )

    kwargs = svc._build_kwargs("hello")

    assert kwargs["provider"] == "openai"
    assert kwargs["prompt"] == "hello"
    assert kwargs["timeout"] == 60