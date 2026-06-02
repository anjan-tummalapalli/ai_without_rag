"""
ai_cli.core.resilience

Resilience utilities:
- Cache (memory + optional Redis)
- RetryEngine
- AsyncRetryEngine
- CircuitBreaker
- RateLimiter
- StreamConsumer
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Callable

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None


class Cache:
    """
    LRU cache with optional Redis backend.
    """

    def __init__(
        self,
        max_entries: int = 1000,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "ai_gateway:",
    ) -> None:
        self.max_entries = max_entries
        self.prefix = prefix

        self._memory: OrderedDict[str, Any] = OrderedDict()
        self._redis = None

        if redis is not None:
            try:
                client = redis.Redis.from_url(redis_url)
                client.ping()
                self._redis = client
            except Exception:
                self._redis = None

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Any:
        if self._redis is not None:
            return self._redis.get(self._key(key))

        if key not in self._memory:
            return None

        value = self._memory.pop(key)
        self._memory[key] = value
        return value

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        if self._redis is not None:
            self._redis.set(
                self._key(key),
                value,
                ex=ttl,
            )
            return

        if key in self._memory:
            self._memory.pop(key)

        self._memory[key] = value

        while len(self._memory) > self.max_entries:
            self._memory.popitem(last=False)

    def delete(self, key: str) -> None:
        if self._redis is not None:
            self._redis.delete(self._key(key))
            return

        self._memory.pop(key, None)

    def clear(self) -> None:
        if self._redis is not None:
            return

        self._memory.clear()


class RetryEngine:
    """
    Synchronous retry executor.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.1,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        if inspect.iscoroutinefunction(func):
            raise TypeError(
                "Coroutine functions are not supported. "
                "Use AsyncRetryEngine instead."
            )

        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)

            except Exception as exc:
                last_exception = exc

                if attempt == self.max_attempts - 1:
                    raise

                time.sleep(self.base_delay * (2**attempt))

        raise last_exception


class AsyncRetryEngine:
    """
    Async retry executor.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 0.1,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def execute(
        self,
        func: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)

            except Exception as exc:
                last_exception = exc

                if attempt == self.max_attempts - 1:
                    raise

                await asyncio.sleep(
                    self.base_delay * (2**attempt)
                )

        raise last_exception


class CircuitBreaker:
    """
    Simple circuit breaker implementation.
    """

    def __init__(
        self,
        threshold: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.threshold = threshold
        self.timeout = timeout

        self.failures = 0
        self.opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True

        elapsed = time.time() - self.opened_at

        if elapsed >= self.timeout:
            return True

        return False

    def wrap(self, func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.allow():
                    raise RuntimeError("circuit open")

                try:
                    result = await func(*args, **kwargs)

                    self.failures = 0
                    self.opened_at = None

                    return result

                except Exception:
                    self.failures += 1

                    if self.failures >= self.threshold:
                        self.opened_at = time.time()

                    raise

            return async_wrapper

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow():
                raise RuntimeError("circuit open")

            try:
                result = func(*args, **kwargs)

                self.failures = 0
                self.opened_at = None

                return result

            except Exception:
                self.failures += 1

                if self.failures >= self.threshold:
                    self.opened_at = time.time()

                raise

        return wrapper


class RateLimiter:
    """
    Token bucket rate limiter.
    """

    def __init__(
        self,
        capacity: int,
        rate_per_second: float,
    ) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)

        self.rate_per_second = rate_per_second
        self.updated_at = time.time()

    def _refill(self) -> None:
        now = time.time()

        elapsed = now - self.updated_at
        self.updated_at = now

        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate_per_second,
        )

    def allow(self, amount: float = 1.0) -> bool:
        self._refill()

        if self.tokens >= amount:
            self.tokens -= amount
            return True

        return False

    def wrap(self, func):
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.allow():
                    raise RuntimeError("rate limit exceeded")

                return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow():
                raise RuntimeError("rate limit exceeded")

            return func(*args, **kwargs)

        return wrapper


class StreamConsumer:
    """
    Background streaming wrapper around provider.send().
    """

    def __init__(self, provider) -> None:
        self.provider = provider

    def stream(
        self,
        prompt: str,
        callback: Callable[[str], None],
    ) -> None:
        def worker():
            try:
                result = self.provider.send(prompt)

                for token in str(result).split():
                    callback(token + " ")

            except Exception as exc:
                callback(f"[ERROR: {exc}]")

        threading.Thread(
            target=worker,
            daemon=True,
        ).start()