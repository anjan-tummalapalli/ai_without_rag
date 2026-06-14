import asyncio
import inspect
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any


# Cache: TTL + thread-safety (memory)
class Cache:
    def __init__(self, max_entries: int = 1000, redis_url: str = "redis://localhost:6379/0", prefix: str = "ai_gateway:"):
        self.max_entries = max_entries
        self.prefix = prefix
        self._memory: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()  # key -> (value, expires_at)
        self._lock = threading.RLock()
        self._redis = None
        try:
            import redis as _redis  # type: ignore
            client = _redis.Redis.from_url(redis_url)
            client.ping()
            self._redis = client
        except Exception:
            self._redis = None

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _purge_expired_locked(self) -> None:
        # Assumes _lock held
        now = time.monotonic()
        keys_to_delete = []
        for k, (_, expires_at) in self._memory.items():
            if expires_at is not None and expires_at <= now:
                keys_to_delete.append(k)
            else:
                # OrderedDict is LRU; stop early if not expired and insertion order preserved
                # But we cannot rely fully, so continue scanning.
                continue
        for k in keys_to_delete:
            self._memory.pop(k, None)

    def get(self, key: str) -> Any:
        if self._redis is not None:
            return self._redis.get(self._key(key))
        with self._lock:
            self._purge_expired_locked()
            item = self._memory.pop(key, None)
            if item is None:
                return None
            value, expires_at = item
            # refresh LRU position
            self._memory[key] = (value, expires_at)
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        if self._redis is not None:
            self._redis.set(self._key(key), value, ex=ttl)
            return
        expires_at = None if ttl is None else time.monotonic() + ttl
        with self._lock:
            self._memory.pop(key, None)
            self._memory[key] = (value, expires_at)
            while len(self._memory) > self.max_entries:
                self._memory.popitem(last=False)

    def delete(self, key: str) -> None:
        if self._redis is not None:
            self._redis.delete(self._key(key))
            return
        with self._lock:
            self._memory.pop(key, None)

    def clear(self) -> None:
        if self._redis is not None:
            return
        with self._lock:
            self._memory.clear()


# RetryEngine with jitter, max_delay, retry_on filtering, and decorator helper
class RetryEngine:
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.1, max_delay: float = 10.0, jitter: bool = True, retry_on: Iterable[type] | None = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = tuple(retry_on) if retry_on is not None else None

    def _should_retry_for(self, exc: Exception) -> bool:
        if self.retry_on is None:
            return True
        return isinstance(exc, self.retry_on)

    def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        if inspect.iscoroutinefunction(func):
            raise TypeError("Coroutine functions are not supported. Use AsyncRetryEngine instead.")
        last_exc = None
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._should_retry_for(exc) or attempt == self.max_attempts - 1:
                    raise
                delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                if self.jitter:
                    delay = random.uniform(0, delay)
                time.sleep(delay)
        raise last_exc

    def decorator(self, *dargs, **dkwargs):
        def _wrap(fn):
            @wraps(fn)
            def _wrapped(*args, **kwargs):
                return self.execute(fn, *args, **kwargs)
            return _wrapped
        return _wrap


# AsyncRetryEngine with the same options
class AsyncRetryEngine:
    def __init__(self, max_attempts: int = 3, base_delay: float = 0.1, max_delay: float = 10.0, jitter: bool = True, retry_on: Iterable[type] | None = None):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_on = tuple(retry_on) if retry_on is not None else None

    def _should_retry_for(self, exc: Exception) -> bool:
        if self.retry_on is None:
            return True
        return isinstance(exc, self.retry_on)

    async def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        last_exc = None
        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if not self._should_retry_for(exc) or attempt == self.max_attempts - 1:
                    raise
                delay = min(self.max_delay, self.base_delay * (2 ** attempt))
                if self.jitter:
                    delay = random.uniform(0, delay)
                await asyncio.sleep(delay)
        raise last_exc

    def decorator(self):
        def _wrap(fn):
            if not asyncio.iscoroutinefunction(fn):
                raise TypeError("decorator for async functions only")
            @wraps(fn)
            async def _wrapped(*args, **kwargs):
                return await self.execute(fn, *args, **kwargs)
            return _wrapped
        return _wrap


# CircuitBreaker with explicit states and thread-safety
class CircuitBreaker:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 5, timeout: float = 60.0, half_open_max_calls: int = 1):
        self.threshold = threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_successes = 0
        self._half_open_calls = 0
        self._lock = threading.RLock()

    def _now(self) -> float:
        return time.monotonic()

    def allow(self) -> bool:
        with self._lock:
            if self._state == self.OPEN:
                assert self._opened_at is not None
                if self._now() - self._opened_at >= self.timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    self._half_open_successes = 0
                    return True
                return False
            return True

    def _record_success(self) -> None:
        with self._lock:
            if self._state in (self.OPEN,):
                return
            if self._state == self.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max_calls:
                    self._state = self.CLOSED
                    self._failures = 0
                    self._opened_at = None
                return
            # CLOSED
            self._failures = 0

    def _record_failure(self) -> None:
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                self._opened_at = self._now()
                self._failures = 1
                return
            self._failures += 1
            if self._failures >= self.threshold:
                self._state = self.OPEN
                self._opened_at = self._now()

    def wrap(self, func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def _async_wrapped(*args, **kwargs):
                if not self.allow():
                    raise RuntimeError("circuit open")
                try:
                    res = await func(*args, **kwargs)
                    self._record_success()
                    return res
                except Exception:
                    self._record_failure()
                    raise
            return _async_wrapped

        @wraps(func)
        def _wrapped(*args, **kwargs):
            if not self.allow():
                raise RuntimeError("circuit open")
            try:
                res = func(*args, **kwargs)
                self._record_success()
                return res
            except Exception:
                self._record_failure()
                raise
        return _wrapped


# RateLimiter: thread-safe and async wait-acquire
class RateLimiter:
    def __init__(self, capacity: int, rate_per_second: float):
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self.rate_per_second = float(rate_per_second)
        self._updated_at = time.monotonic()
        self._lock = threading.RLock()

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._updated_at = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_per_second)

    def allow(self, amount: float = 1.0) -> bool:
        with self._lock:
            self._refill_locked()
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    async def acquire(self, amount: float = 1.0, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            await asyncio.sleep(max(0.001, 1.0 / max(1.0, self.rate_per_second)))
