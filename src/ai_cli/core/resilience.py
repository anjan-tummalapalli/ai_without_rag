from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def execute_with_fallback(primary_fn: Callable[..., T], fallback_fn: Callable[..., T], *args, **kwargs) -> T:
    """
    Execute primary_fn; if it raises, execute fallback_fn.
    """
    try:
        return primary_fn(*args, **kwargs)
    except Exception:
        return fallback_fn(*args, **kwargs)


class Cache:
    def __init__(self, max_entries: int = 1000, prefix: str = "ai_gateway:"):
        self.max_entries = max_entries
        self.prefix = prefix
        self._memory: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = threading.RLock()

    def _purge_expired_locked(self) -> None:
        now = time.monotonic()
        keys_to_delete = []
        for k, (_, expires_at) in list(self._memory.items()):
            if expires_at is not None and expires_at <= now:
                keys_to_delete.append(k)
        for k in keys_to_delete:
            self._memory.pop(k, None)

    def get(self, key: str) -> Any:
        with self._lock:
            self._purge_expired_locked()
            item = self._memory.pop(key, None)
            if item is None:
                return None
            value, expires_at = item
            self._memory[key] = (value, expires_at)
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expires_at = None if ttl is None else time.monotonic() + ttl
        with self._lock:
            self._memory.pop(key, None)
            self._memory[key] = (value, expires_at)
            while len(self._memory) > self.max_entries:
                self._memory.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._memory.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._memory.clear()


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

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            if self._state == self.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_max_calls:
                    self._state = self.CLOSED

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold:
                self._state = self.OPEN
                self._opened_at = self._now()