import pytest
from ai_cli.core.resilience import (
    execute_with_fallback,
    RateLimiter,
    CircuitBreaker,
    RetryEngine,
    AsyncRetryEngine,
    Cache,
)
from ai_cli.core.resilience import execute_with_fallback
from ai_cli.core.resilience import Cache
from ai_cli.core.resilience import RetryEngine


def test_execute_primary_success():
    result = execute_with_fallback(
        lambda: "success",
        fallback_fn=lambda: "fallback",
    )

    assert result == "success"

def test_execute_fallback_on_failure():
    result = execute_with_fallback(
        lambda: 1 / 0,
        fallback_fn=lambda: "fallback",
    )

    assert result == "fallback"

def test_rate_limiter():
    limiter = RateLimiter(
        capacity=1,
        rate_per_second=0
    )

    assert limiter.allow()
    assert not limiter.allow()


def test_circuit_breaker_opens():
    cb = CircuitBreaker(threshold=1)
    wrapped = cb.wrap(lambda: 1/0)
    try:
        wrapped()
    except ZeroDivisionError:
        pass

    assert cb.allow() is False

def test_cache_memory_operations():
    cache = Cache(max_entries=2)
    assert cache.get("missing") is None
    cache.set("a", "value")
    assert cache.get("a") == "value"
    cache.delete("a")
    assert cache.get("a") is None
    cache.set("x", 1)
    cache.set("y", 2)
    cache.clear()
    assert cache.get("x") is None

def test_retry_engine_success():
    engine = RetryEngine(max_attempts=2)
    assert engine.execute(lambda: "ok") == "ok"

def test_retry_engine_retry_then_success(monkeypatch):
    engine = RetryEngine(
        max_attempts=2,
        base_delay=0,
        jitter=False,
    )
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("fail")
        return "ok"
    assert engine.execute(flaky) == "ok"

def test_retry_engine_failure():
    engine = RetryEngine(
        max_attempts=1
    )

    with pytest.raises(ValueError):
        engine.execute(lambda: (_ for _ in ()).throw(ValueError()))

@pytest.mark.asyncio
async def test_rate_limiter_acquire_timeout():
    limiter = RateLimiter(
        capacity=0,
        rate_per_second=1,
    )
    result = await limiter.acquire(
        timeout=0.01
    )
    assert result is False

def test_circuit_breaker_reopens():
    cb = CircuitBreaker(
        threshold=1,
        timeout=0,
    )

    wrapped = cb.wrap(
        lambda: (_ for _ in ()).throw(RuntimeError())
    )

    with pytest.raises(RuntimeError):
        wrapped()

    assert cb.allow() is True

@pytest.mark.asyncio
async def test_async_retry_engine_success():
    from ai_cli.core.resilience import AsyncRetryEngine

    engine = AsyncRetryEngine(
        max_attempts=2,
        base_delay=0,
        jitter=False,
    )

    async def ok():
        return "done"

    result = await engine.execute(ok)

    assert result == "done"


@pytest.mark.asyncio
async def test_async_retry_engine_failure():
    from ai_cli.core.resilience import AsyncRetryEngine

    engine = AsyncRetryEngine(
        max_attempts=1,
    )

    async def fail():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        await engine.execute(fail)

def test_circuit_breaker_success():
    from ai_cli.core.resilience import CircuitBreaker

    cb = CircuitBreaker()

    wrapped = cb.wrap(lambda: "ok")

    assert wrapped() == "ok"

def test_retry_engine_decorator():
    from ai_cli.core.resilience import RetryEngine

    engine = RetryEngine(
        max_attempts=1
    )

    @engine.decorator()
    def hello():
        return "hello"

    assert hello() == "hello"

def test_cache_expiry():
    import time
    from ai_cli.core.resilience import Cache

    cache = Cache()

    cache.set(
        "temp",
        "value",
        ttl=0.01
    )

    time.sleep(0.02)

    assert cache.get("temp") is None

def test_cache_delete_clear():
    from ai_cli.core.resilience import Cache

    cache = Cache()
    cache.set("a", 1)
    assert cache.get("a") == 1
    cache.delete("a")
    assert cache.get("a") is None
    cache.set("b", 2)
    cache.clear()
    assert cache.get("b") is None

def test_retry_engine_rejects_async():
    from ai_cli.core.resilience import RetryEngine

    engine = RetryEngine()
    async def async_fn():
        return "x"

    with pytest.raises(TypeError):
        engine.execute(async_fn)

def test_circuit_breaker_failure():
    from ai_cli.core.resilience import CircuitBreaker
    cb = CircuitBreaker(threshold=1)
    wrapped = cb.wrap(
        lambda: 1 / 0
    )

    with pytest.raises(ZeroDivisionError):
        wrapped()

    with pytest.raises(RuntimeError):
        wrapped()

def test_rate_limiter_refill():
    import time
    from ai_cli.core.resilience import RateLimiter
    limiter = RateLimiter(
        capacity=1,
        rate_per_second=100
    )
    assert limiter.allow()
    time.sleep(0.02)
    assert limiter.allow()

def test_async_retry_decorator_rejects_sync():
    from ai_cli.core.resilience import AsyncRetryEngine
    engine = AsyncRetryEngine()
    with pytest.raises(TypeError):
        @engine.decorator()
        def normal():
            return "x"

def test_cache_without_redis():
    from ai_cli.core.resilience import Cache

    cache = Cache(
        redis_url="redis://invalid-host:9999"
    )
    cache.set("key", "value")
    assert cache.get("key") == "value"

def test_retry_engine_exhausted():
    from ai_cli.core.resilience import RetryEngine

    engine = RetryEngine(
        max_attempts=2,
        base_delay=0,
        jitter=False
    )

    def fail():
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        engine.execute(fail)

@pytest.mark.asyncio
async def test_async_retry_decorator():
    from ai_cli.core.resilience import AsyncRetryEngine

    engine = AsyncRetryEngine()

    @engine.decorator()
    async def hello():
        return "ok"

    assert await hello() == "ok"

def test_circuit_breaker_half_open():
    import time

    from ai_cli.core.resilience import CircuitBreaker

    cb = CircuitBreaker(
        threshold=1,
        timeout=0
    )
    cb._record_failure()
    assert cb.allow()
    cb._record_success()
    assert cb.allow()

@pytest.mark.asyncio
async def test_rate_limiter_acquire_success():
    from ai_cli.core.resilience import RateLimiter

    limiter = RateLimiter(
        capacity=1,
        rate_per_second=100
    )

    assert await limiter.acquire()

def test_execute_with_fallback_exception_path():
    from ai_cli.core.resilience import execute_with_fallback

    result = execute_with_fallback(
        lambda: (_ for _ in ()).throw(ValueError()),
        lambda: "fallback",
    )

    assert result == "fallback"

def test_cache_eviction():

    from ai_cli.core.resilience import Cache

    cache = Cache(max_entries=1)

    cache.set("a", 1)
    cache.set("b", 2)

    assert cache.get("a") is None
    assert cache.get("b") == 2

def test_retry_engine_decorator_executes():

    from ai_cli.core.resilience import RetryEngine

    engine = RetryEngine()

    @engine.decorator()
    def hello():
        return "ok"

    assert hello() == "ok"

def test_retry_engine_retry_filter():

    from ai_cli.core.resilience import RetryEngine

    engine = RetryEngine(
        max_attempts=2,
        retry_on=(ValueError,),
        base_delay=0
    )

    def fail():
        raise RuntimeError()

    with pytest.raises(RuntimeError):
        engine.execute(fail)

@pytest.mark.asyncio
async def test_circuit_breaker_async_wrap():

    from ai_cli.core.resilience import CircuitBreaker

    cb = CircuitBreaker()

    @cb.wrap
    async def hello():
        return "ok"

    assert await hello() == "ok"

def test_rate_limiter_constructor_values():

    from ai_cli.core.resilience import RateLimiter

    limiter = RateLimiter(
        capacity=5,
        rate_per_second=2
    )

    assert limiter.capacity == 5.0
    assert limiter.rate_per_second == 2.0

def test_retry_engine_rejects_coroutine():
    async def fn():
        return 1

    engine = RetryEngine()

    with pytest.raises(TypeError):
        engine.execute(fn)

def test_async_retry_decorator_rejects_sync():
    engine = AsyncRetryEngine()

    def normal():
        return 1

    with pytest.raises(TypeError):
        engine.decorator()(normal)

@pytest.mark.asyncio
async def test_circuit_breaker_async_wrap():

    cb = CircuitBreaker(threshold=1)

    @cb.wrap
    async def ok():
        return "ok"

    assert await ok()


