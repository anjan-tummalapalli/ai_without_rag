from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

P = ParamSpec("P")
R = TypeVar("R")


class RateLimiter:
    def __init__(
        self,
        capacity: int = 1,
        rate_per_second: float = 1.0,
        period: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.capacity = capacity
        self.rate_per_second = rate_per_second
        self.period = (
            period / rate_per_second if rate_per_second > 0 else period
        )
        self.calls: deque[float] = deque()

    def allow(self) -> bool:
        now = time.monotonic()
        while self.calls and (now - self.calls[0] >= self.period):
            self.calls.popleft()

        if len(self.calls) >= self.capacity:
            return False

        self.calls.append(now)
        return True

    async def acquire(self, timeout: float | None = None) -> bool:
        start = time.monotonic()
        while not self.allow():
            if timeout is not None and time.monotonic() - start >= timeout:
                return False
            await asyncio.sleep(0.01)
        return True


class CircuitBreaker:
    def __init__(
        self,
        threshold: int = 3,
        timeout: float = 60,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.threshold = (
            failure_threshold if failure_threshold is not None else threshold
        )
        self.recovery_timeout = (
            recovery_timeout if recovery_timeout is not None else timeout
        )
        self.failure_count = 0
        self.open = False
        self.last_failure_time: float | None = None

    def record_success(self) -> None:
        self.failure_count = 0
        self.open = False

    def _record_success(self) -> None:
        self.record_success()

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.threshold:
            self.open = True
            self.last_failure_time = time.monotonic()

    def _record_failure(self) -> None:
        self.record_failure()

    @overload
    def wrap(
        self, func: Callable[P, Awaitable[R]]
    ) -> Callable[P, Awaitable[R]]: ...

    @overload
    def wrap(self, func: Callable[P, R]) -> Callable[P, R]: ...

    def wrap(self, func: Callable[P, Any]) -> Callable[P, Any]:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                if not self.allow():
                    raise RuntimeError("Circuit open")
                try:
                    result = await cast(Callable[P, Awaitable[Any]], func)(
                        *args, **kwargs
                    )
                    self.record_success()
                    return result
                except Exception:
                    self.record_failure()
                    raise

            return async_wrapper

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            if not self.allow():
                raise RuntimeError("Circuit open")
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise

        return wrapper

    def allow(self) -> bool:
        if not self.open:
            return True

        if (
            self.last_failure_time is not None
            and time.monotonic() - self.last_failure_time
            > self.recovery_timeout
        ):
            self.open = False
            return True

        return False


class RetryEngine:
    def __init__(
        self,
        max_attempts: int = 3,
        retries: int | None = None,
        base_delay: float = 0,
        retry_on: Callable[[Exception], bool]
        | tuple[type[Exception], ...]
        | None = None,
        retry_filter: Callable[[Exception], bool]
        | tuple[type[Exception], ...]
        | None = None,
        **kwargs: Any,
    ) -> None:
        self.max_attempts = retries or max_attempts
        self.base_delay = base_delay
        self.retry_on = retry_on or retry_filter

    def execute(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        if inspect.iscoroutinefunction(func):
            raise TypeError("RetryEngine cannot execute async functions")

        last: Exception | None = None

        for _ in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last = exc

                if self.retry_on:
                    if isinstance(self.retry_on, tuple):
                        if not isinstance(exc, self.retry_on):
                            break
                    elif not self.retry_on(exc):
                        break

                if self.base_delay:
                    time.sleep(self.base_delay)

        if last is None:
            raise RuntimeError("RetryEngine failed without exception")
        raise last

    def run(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        return self.execute(func, *args, **kwargs)

    def decorator(self) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def deco(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return self.execute(func, *args, **kwargs)

            return wrapper

        return deco


class AsyncRetryEngine:
    def __init__(
        self,
        max_attempts: int = 3,
        retries: int | None = None,
        base_delay: float = 0,
        **kwargs: Any,
    ) -> None:
        self.max_attempts = retries or max_attempts
        self.base_delay = base_delay

    def decorator(
        self,
    ) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
        def deco(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
            if not inspect.iscoroutinefunction(func):
                raise TypeError("AsyncRetryEngine requires async function")

            @wraps(func)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last: Exception | None = None

                for _ in range(self.max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        last = exc

                if last is None:
                    raise RuntimeError(
                        "AsyncRetryEngine failed without exception"
                    )
                raise last

            return wrapper

        return deco

    def __call__(
        self, func: Callable[P, Awaitable[R]]
    ) -> Callable[P, Awaitable[R]]:
        return self.decorator()(func)

    async def execute(
        self, func: Callable[P, Awaitable[R]], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        if not inspect.iscoroutinefunction(func):
            raise TypeError("AsyncRetryEngine requires async function")

        last: Exception | None = None

        for _ in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last = exc

        if last is None:
            raise RuntimeError("AsyncRetryEngine failed without exception")
        raise last


class Cache:
    def __init__(
        self,
        max_entries: int = 100,
        redis_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.max_entries = max_entries
        self.data: dict[Any, tuple[Any, float | None]] = {}

    def set(self, key: Any, value: Any, ttl: float | None = None) -> None:
        expiry = time.monotonic() + ttl if ttl else None

        if len(self.data) >= self.max_entries:
            oldest = next(iter(self.data))
            del self.data[oldest]

        self.data[key] = (value, expiry)

    def get(self, key: Any, default: Any = None) -> Any:
        if key not in self.data:
            return default

        value, expiry = self.data[key]

        if expiry is not None and time.monotonic() > expiry:
            del self.data[key]
            return default

        return value

    def delete(self, key: Any) -> None:
        self.data.pop(key, None)

    def clear(self) -> None:
        self.data.clear()


def execute_with_fallback(
    primary: Callable[P, R],
    fallback: Callable[P, R] | None = None,
    fallback_fn: Callable[P, R] | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> R | None:
    if fallback is None:
        fallback = fallback_fn

    try:
        return primary(*args, **kwargs)
    except Exception:
        if fallback is not None:
            return fallback(*args, **kwargs)
        return None


__all__ = [
    "RateLimiter",
    "CircuitBreaker",
    "RetryEngine",
    "AsyncRetryEngine",
    "Cache",
    "execute_with_fallback",
]
