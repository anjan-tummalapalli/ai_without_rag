from __future__ import annotations
import asyncio
import functools
import inspect
import os
import threading
import time
import random
import logging
from collections import deque
from typing import Any, Callable, Deque, Optional, TypeVar

try:
    import redis
except Exception:  # optional
    redis = None  # type: ignore

T = TypeVar("T")


class Cache:
    """Simple cache: Redis-backed when available, else in-memory LRU."""

    def __init__(self, namespace: str = "ai_gateway", max_entries: int = 1024):
        self.namespace = namespace
        self.max_entries = max_entries
        self._local_cache: dict[str, Any] = {}
        self._local_order: Deque[str] = deque()
        if redis is not None:
            try:
                url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                self._redis = redis.Redis.from_url(url)
                self._redis.ping()
            except Exception:
                self._redis = None
        else:
            self._redis = None

    def _key(self, k: str) -> str:
        return f"{self.namespace}:{k}"

    def get(self, key: str) -> Optional[Any]:
        if self._redis:
            try:
                val = self._redis.get(self._key(key))
                return None if val is None else val
            except Exception:
                return None
        return self._local_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self._redis:
            try:
                self._redis.set(self._key(key), value, ex=ttl)
                return
            except Exception:
                pass
        if key in self._local_cache:
            try:
                self._local_order.remove(key)
            except ValueError:
                pass
        self._local_cache[key] = value
        self._local_order.appendleft(key)
        while len(self._local_order) > self.max_entries:
            old = self._local_order.pop()
            self._local_cache.pop(old, None)


GLOBAL_CACHE = Cache()


class AsyncRetryEngine:
    """Async retry executor with exponential backoff."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 0.5):
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    async def execute(self, coro_fn: Callable[[], Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                res = coro_fn()
                if inspect.isawaitable(res):
                    return await res
                return res
            except Exception as exc:
                last_exc = exc
                sleep_time = self.base_delay * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time + (0.1 * attempt))
        assert last_exc is not None
        raise last_exc

logger = logging.getLogger("ai_gateway")


class RetryEngine:
    """Sync retry executor with exponential backoff."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def execute(self, func: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return func()
            except Exception as exc:
                last_error = exc
                sleep_time = self.base_delay * (2 ** (attempt - 1))
                jitter = random.uniform(0, 0.5)
                logger.warning(
                    "retry_attempt=%s sleep=%s error=%s",
                    attempt,
                    sleep_time,
                    exc,
                )
                time.sleep(sleep_time + jitter)
        raise last_error  # type: ignore[misc]


class CircuitBreaker:
    """Minimal thread-safe circuit breaker."""

    def __init__(self, threshold: int = 5, timeout: int = 30):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.open_until = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            return time.time() >= self.open_until

    def success(self) -> None:
        with self._lock:
            self.failures = 0
            self.open_until = 0.0

    def failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold:
                self.open_until = time.time() + self.timeout

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            if not self.allow():
                raise RuntimeError("circuit open")
            try:
                result = fn(*args, **kwargs)
            except Exception:
                self.failure()
                raise
            else:
                self.success()
                return result

        return _wrapped


class RateLimiter:
    """Token-bucket rate limiter."""

    def __init__(self, capacity: int = 10, rate_per_second: float = 1.0):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.rate = rate_per_second
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        add = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + add)

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            self._refill()
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

    def wrap(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            if not self.allow():
                raise RuntimeError("rate limited")
            return fn(*args, **kwargs)

        return _wrapped


class StreamConsumer:
    """Helper to adapt synchronous providers into streaming-like behavior."""

    def __init__(self, provider: Any):
        self.provider = provider

    def stream(self, prompt: str, on_token: Callable[[str], None]) -> None:
        if hasattr(self.provider, "send_stream"):
            return self.provider.send_stream(prompt, on_token)

        def _worker():
            resp = self.provider.send(prompt)
            for token in resp.split():
                on_token(token + " ")
            on_token("")  # sentinel

        thr = threading.Thread(target=_worker, daemon=True)
        thr.start()