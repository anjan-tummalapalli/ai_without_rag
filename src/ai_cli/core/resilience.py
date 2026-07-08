import asyncio
import inspect
import time
from collections import deque


class RateLimiter:
    def __init__(
        self,
        capacity=1,
        rate_per_second=1.0,
        period=1.0,
        **kwargs,
    ):
        self.capacity = capacity
        self.rate_per_second = rate_per_second

        # important:
        self.period = (
            period / rate_per_second if rate_per_second > 0 else period
        )

        self.calls = deque()

    def allow(self):
        now = time.monotonic()

        while self.calls and (now - self.calls[0] >= self.period):
            self.calls.popleft()

        if len(self.calls) >= self.capacity:
            return False

        self.calls.append(now)
        return True

    async def acquire(self, timeout=None):
        start = time.monotonic()

        while not self.allow():
            if timeout is not None and time.monotonic() - start >= timeout:
                return False
            await asyncio.sleep(0.01)

        return True


class CircuitBreaker:
    def __init__(
        self,
        threshold=3,
        timeout=60,
        failure_threshold=None,
        recovery_timeout=None,
        **kwargs,
    ):
        self.threshold = (
            failure_threshold if failure_threshold is not None else threshold
        )

        self.recovery_timeout = (
            recovery_timeout if recovery_timeout is not None else timeout
        )

        self.failure_count = 0
        self.open = False
        self.last_failure_time = None

    def record_success(self):
        self.failure_count = 0
        self.open = False

    def _record_success(self):
        self.record_success()

    def record_failure(self):
        self.failure_count += 1

        if self.failure_count >= self.threshold:
            self.open = True
            self.last_failure_time = time.monotonic()

    def _record_failure(self):
        self.record_failure()

    def wrap(self, func):
        if inspect.iscoroutinefunction(func):

            async def async_wrapper(*args, **kwargs):
                if not self.allow():
                    raise RuntimeError("Circuit open")

                try:
                    result = await func(*args, **kwargs)
                    self.record_success()
                    return result

                except Exception:
                    self.record_failure()
                    raise

            return async_wrapper

        def wrapper(*args, **kwargs):
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

    def allow(self):
        if not self.open:
            return True

        if (
            self.last_failure_time
            and time.monotonic() - self.last_failure_time
            > self.recovery_timeout
        ):
            self.open = False
            return True

        return False


class RetryEngine:
    def __init__(
        self,
        max_attempts=3,
        retries=None,
        base_delay=0,
        retry_on=None,
        retry_filter=None,
        **kwargs,
    ):
        self.max_attempts = retries or max_attempts
        self.base_delay = base_delay
        self.retry_on = retry_on or retry_filter

    def execute(self, func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            raise TypeError("RetryEngine cannot execute async functions")

        last = None

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
        raise last

    def run(self, func, *args, **kwargs):
        return self.execute(func, *args, **kwargs)

    def decorator(self):

        def deco(func):

            def wrapper(*args, **kwargs):
                return self.execute(
                    func,
                    *args,
                    **kwargs,
                )

            return wrapper

        return deco


class AsyncRetryEngine:
    def __init__(
        self,
        max_attempts=3,
        retries=None,
        base_delay=0,
        **kwargs,
    ):
        self.max_attempts = retries or max_attempts
        self.base_delay = base_delay

    def decorator(self):

        def deco(func):

            if not inspect.iscoroutinefunction(func):
                raise TypeError("AsyncRetryEngine requires async function")

            async def wrapper(*args, **kwargs):
                last = None

                for _ in range(self.max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        last = exc

                raise last

            return wrapper

        return deco

    def __call__(self, func):
        return self.decorator()(func)

    async def execute(self, func, *args, **kwargs):
        if not inspect.iscoroutinefunction(func):
            raise TypeError("AsyncRetryEngine requires async function")

        last = None

        for _ in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last = exc

        raise last


class Cache:
    def __init__(
        self,
        max_entries=100,
        redis_url=None,
        **kwargs,
    ):
        self.max_entries = max_entries
        self.data = {}

    def set(self, key, value, ttl=None):
        expiry = time.monotonic() + ttl if ttl else None

        if len(self.data) >= self.max_entries:
            oldest = next(iter(self.data))
            del self.data[oldest]

        self.data[key] = (value, expiry)

    def get(self, key, default=None):
        if key not in self.data:
            return default

        value, expiry = self.data[key]

        if expiry and time.monotonic() > expiry:
            del self.data[key]
            return default

        return value

    def delete(self, key):
        self.data.pop(key, None)

    def clear(self):
        self.data.clear()


def execute_with_fallback(  # pylint: disable=keyword-arg-before-vararg
    primary,
    fallback=None,
    fallback_fn=None,
    *args,
    **kwargs,
):
    """Call *primary*, falling back to *fallback*/*fallback_fn* on error.

    ``fallback``/``fallback_fn`` are intentionally positional (callers rely
    on ``execute_with_fallback(primary, fallback)``); ``*args``/``**kwargs``
    are forwarded to whichever of the two ends up being called.
    """
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
