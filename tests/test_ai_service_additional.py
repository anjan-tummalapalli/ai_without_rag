import inspect

import pytest

from ai_cli.core.service.ai_service import AIService


def test_build_kwargs_fallback_includes_optional_fields(monkeypatch):
    """Cover lines 166,168,170,172."""

    monkeypatch.setattr(
        inspect,
        "signature",
        lambda *_: (_ for _ in ()).throw(TypeError()),
    )

    svc = AIService(
        provider="openai",
        model="gpt-4",
        timeout=30,
        profile="dev",
        stream=True,
        modules="a,b",
    )

    kwargs = svc._build_kwargs("hello")

    assert kwargs["provider"] == "openai"
    assert kwargs["prompt"] == "hello"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["profile"] == "dev"
    assert kwargs["stream"] is True
    assert kwargs["modules"] == ["a", "b"]


@pytest.mark.asyncio
async def test_drain_async_with_async_iterable():
    """Cover lines 184-185."""

    class AsyncOnly:
        def __aiter__(self):
            async def gen():
                yield "one"
                yield "two"

            return gen()

    result = await AIService._drain_async(AsyncOnly())

    assert result == "onetwo"


def test_drain_sync_with_awaitable(monkeypatch):
    """Cover line 199."""

    called = {}

    async def fake_drain(result):
        # Consume the coroutine so it is awaited.
        value = await result
        called["hit"] = True
        assert value == "ignored"
        return "done"

    monkeypatch.setattr(AIService, "_drain_async", fake_drain)

    async def coro():
        return "ignored"

    assert AIService._drain_sync(coro()) == "done"
    assert called["hit"] is True


@pytest.mark.asyncio
async def test_ask_async_async_generator_branch(monkeypatch):
    """Cover lines 358,361-362."""

    svc = AIService()

    async def fake_core():
        async def gen():
            yield "A"
            yield "B"

        return gen()

    monkeypatch.setattr(
        svc,
        "_call_with_retries",
        lambda prompt: fake_core(),
    )

    parts = []

    async for chunk in svc.ask_async("hello"):
        parts.append(chunk)

    assert parts == ["A", "B"]