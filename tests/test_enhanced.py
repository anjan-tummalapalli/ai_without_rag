from __future__ import annotations
import os
import time
import pytest
import tempfile
from unittest.mock import MagicMock, patch
from ai_cli.core.prompt_corrector import PromptCorrector, prompt_corrector
from ai_cli.core.exceptions import (
    PromptValidationError,
    ProviderRequestError,
    ResponseValidationError,
)
from ai_cli.core.resilience import (
    Cache,
    RetryEngine,
    AsyncRetryEngine,
    CircuitBreaker,
    RateLimiter,
    StreamConsumer,
)
from ai_cli.providers.base import EchoProvider, AIProvider, ProviderMetadata
from ai_cli.providers.auto_provider import AutoProvider
from ai_cli.providers.registry import PROVIDER_MAP
from ai_cli.utils.secrets import SecretManager, is_kubernetes
from ai_cli.utils.validation import HallucinationDetector, ResponseValidator
from ai_cli.telemetry.monitoring import ModelQualityMetrics
from ai_cli.cli import build_parser, _build_ask_kwargs, _decode_chunk


# =============================================================================
# 1. PromptCorrector Tests
# =============================================================================
def test_prompt_corrector_basic() -> None:
    # Test whitespace collapsing
    assert prompt_corrector.correct("Hello   world  !") == "Hello world!"
    # Test typo correction
    assert prompt_corrector.correct("Teh government is weird.") == "The government is weird."
    assert prompt_corrector.correct("dont do that") == "don't do that"
    # Test capitalization preservation in typo map
    assert prompt_corrector.correct("TEH government") == "THE government"
    assert prompt_corrector.correct("Teh government") == "The government"
    # Test punctuation spacing
    assert prompt_corrector.correct("Hello ? What , is this ?") == "Hello? What, is this?"
    assert prompt_corrector.correct("yes,it is.") == "yes, it is."


def test_prompt_corrector_brackets_and_quotes() -> None:
    # Test bracket balancing
    assert prompt_corrector.correct("Hello (world") == "Hello (world)"
    assert prompt_corrector.correct("Data [nested (bracket}") == "Data [nested (bracket)]"
    # Test quote balancing
    assert prompt_corrector.correct("She said \"hello") == "She said \"hello\""
    assert prompt_corrector.correct("It's a user's choice") == "It's a user's choice"
    # Single quotes balancing check
    assert prompt_corrector.correct("This 'is unbalanced") == "This 'is unbalanced'"


def test_prompt_corrector_control_chars() -> None:
    # Test NUL byte and control character removal
    assert prompt_corrector.correct("Hello\x00world\x03!") == "Helloworld!"


def test_prompt_corrector_custom_config() -> None:
    custom = PromptCorrector(
        typo_map={"foo": "bar"},
        collapse_spaces=False,
        fix_punctuation=False,
        balance_brackets=False,
        clean_control_chars=False,
    )
    # Spacing and brackets shouldn't change, typo should
    text = "foo    hello (world \x00 ?"
    assert custom.correct(text) == "bar    hello (world \x00 ?"


def test_prompt_corrector_invalid_inputs() -> None:
    with pytest.raises(PromptValidationError):
        prompt_corrector.correct(None)  # type: ignore
    with pytest.raises(PromptValidationError):
        prompt_corrector.correct(123)  # type: ignore


# =============================================================================
# 2. Resilience Core Tests
# =============================================================================
def test_cache_in_memory() -> None:
    cache = Cache(max_entries=3)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") == 3

    # Evict 'a' (LRU)
    cache.set("d", 4)
    assert cache.get("a") is None
    assert cache.get("d") == 4

    # Update order
    cache.get("b")
    cache.set("e", 5)  # should evict 'c' because 'b' was accessed recently
    assert cache.get("c") is None
    assert cache.get("b") == 2


@patch("ai_cli.core.resilience.redis")
def test_cache_redis_fallback(mock_redis) -> None:
    # Force Redis initialization success to test Redis path
    mock_redis_client = MagicMock()
    mock_redis.Redis.from_url.return_value = mock_redis_client
    mock_redis_client.ping.return_value = True

    cache = Cache()
    assert cache._redis == mock_redis_client

    cache.set("redis_key", "val", ttl=60)
    mock_redis_client.set.assert_called_with("ai_gateway:redis_key", "val", ex=60)

    mock_redis_client.get.return_value = b"redis_val"
    assert cache.get("redis_key") == b"redis_val"


def test_retry_engine_sync() -> None:
    engine = RetryEngine(max_attempts=3, base_delay=0.001)
    # Success on first attempt
    call_count = 0
    def success_func():
        nonlocal call_count
        call_count += 1
        return "ok"
    assert engine.execute(success_func) == "ok"
    assert call_count == 1

    # Failure then success
    call_count = 0
    def fail_once_func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("transient error")
        return "resolved"
    assert engine.execute(fail_once_func) == "resolved"
    assert call_count == 2

    # Persistent failure
    def always_fail():
        raise RuntimeError("fatal error")
    with pytest.raises(RuntimeError) as exc:
        engine.execute(always_fail)
    assert str(exc.value) == "fatal error"


@pytest.mark.asyncio
async def test_async_retry_engine() -> None:
    engine = AsyncRetryEngine(max_attempts=2, base_delay=0.001)

    # Async success
    async def async_success():
        return "async_ok"
    assert await engine.execute(async_success) == "async_ok"

    # Async persistent failure
    async def async_fail():
        raise ZeroDivisionError("division by zero")
    with pytest.raises(ZeroDivisionError):
        await engine.execute(async_fail)


def test_circuit_breaker() -> None:
    cb = CircuitBreaker(threshold=2, timeout=1)
    assert cb.allow() is True

    # First failure
    with pytest.raises(ValueError):
        @cb.wrap
        def fail_fn():
            raise ValueError("fail")
        fail_fn()

    assert cb.allow() is True

    # Second failure triggers circuit open
    with pytest.raises(ValueError):
        fail_fn()

    assert cb.allow() is False
    with pytest.raises(RuntimeError) as exc:
        fail_fn()
    assert "circuit open" in str(exc.value)

    # Wait for timeout to recover (half-open/allow probe)
    time.sleep(1.1)
    assert cb.allow() is True

    # Success closes the circuit completely
    @cb.wrap
    def success_fn():
        return "yay"
    assert success_fn() == "yay"
    assert cb.failures == 0
    assert cb.allow() is True


def test_rate_limiter() -> None:
    limiter = RateLimiter(capacity=2, rate_per_second=10)
    assert limiter.allow(1.0) is True
    assert limiter.allow(1.0) is True
    assert limiter.allow(1.0) is False  # capacity exhausted

    # Test wrap decorator
    @limiter.wrap
    def dummy():
        return "wrapped"

    # Wait for refill
    time.sleep(0.15)
    assert dummy() == "wrapped"


def test_stream_consumer() -> None:
    class DummyProvider:
        def send(self, prompt):
            return "one two three"

    consumer = StreamConsumer(DummyProvider())
    tokens = []
    def on_token(t):
        if t:
            tokens.append(t)

    consumer.stream("hello", on_token)
    # Wait briefly since consumer.stream uses a background daemon Thread
    time.sleep(0.1)
    assert tokens == ["one ", "two ", "three "]


# =============================================================================
# 3. Providers & Plugins Tests
# =============================================================================
def test_echo_provider() -> None:
    provider = EchoProvider()
    assert provider.provider_name == "echo"
    assert provider.model == "echo"
    assert provider.send("test prompt") == "(echo) test prompt"


def test_ai_provider_base_methods() -> None:
    meta = ProviderMetadata(
        name="Test",
        default_model="m1",
        supported_models=["m1"],
        supports_streaming=False,
        supports_tools=False,
        supports_vision=False,
        max_context=1000,
        cost_per_1k_tokens=0.0,
        avg_latency_ms=10,
    )
    provider = AIProvider(provider_name="test", provider_meta=meta)

    # Coerce responses
    # Let's verify coercion exception or handling
    with pytest.raises(ProviderRequestError):
        provider._coerce_response_to_str(None)

    assert provider._coerce_response_to_str("hello") == "hello"
    assert provider._coerce_response_to_str(b"bytes") == "bytes"
    assert provider._coerce_response_to_str({"a": 1}) == '{"a": 1}'

    # Validate prompt (uses PromptCorrector under the hood now)
    assert provider.validate_prompt("  teh   correct   ") == "the correct"
    with pytest.raises(PromptValidationError):
        provider.validate_prompt("")


def test_prompt_corrector_apostrophe_explicit() -> None:
    # Explicitly test word boundary apostrophes vs actual opening/closing single quotes
    assert prompt_corrector.correct("'hello'") == "'hello'"
    assert prompt_corrector.correct("user's guide") == "user's guide"
    assert prompt_corrector.correct("don't fail's") == "don't fail's"


def test_cache_true_lru() -> None:
    cache = Cache(max_entries=3)
    cache.set("x", 10)
    cache.set("y", 20)
    cache.set("z", 30)

    # Access x: makes it most recently used
    assert cache.get("x") == 10

    # set a new key: should evict y (since z was added after x, and x was accessed)
    cache.set("w", 40)
    assert cache.get("y") is None
    assert cache.get("x") == 10
    assert cache.get("z") == 30
    assert cache.get("w") == 40


def test_auto_provider_fallback() -> None:
    # 1. All registered providers fail
    auto = AutoProvider()
    auto.fallback_order = ["non_existent_1", "non_existent_2"]
    with pytest.raises(ProviderRequestError) as exc:
        auto.send("hello")
    assert "Auto fallback exhausted" in str(exc.value)

    # 2. Test successful fallback
    class MockSuccessProvider(AIProvider):
        def __init__(self, model=None):
            meta = ProviderMetadata(
                name="Success",
                default_model="s1",
                supported_models=["s1"],
                supports_streaming=False,
                supports_tools=False,
                supports_vision=False,
                max_context=1000,
                cost_per_1k_tokens=0.0,
                avg_latency_ms=10,
            )
            super().__init__("mock_success", model, provider_meta=meta)
        def _send_impl(self, prompt):
            return "success_resp"

    with patch.dict(PROVIDER_MAP, {"mock_success": MockSuccessProvider}, clear=False):
        auto = AutoProvider()
        auto.fallback_order = ["invalid_prov", "mock_success"]
        assert auto.send("hello") == "success_resp"


# =============================================================================
# 4. Utilities Tests
# =============================================================================
def test_secret_manager() -> None:
    # From environment
    with patch.dict(os.environ, {"MY_TEST_SECRET": "env_value"}):
        assert SecretManager.get_secret("MY_TEST_SECRET") == "env_value"

    # From file fallback
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"file_value\n")
        tmp_path = tmp.name

    try:
        assert SecretManager.get_secret("MISSING_ENV_SECRET", tmp_path) == "file_value"
    finally:
        os.remove(tmp_path)

    # Env takes precedence
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"file_value")
        tmp_path = tmp.name

    try:
        with patch.dict(os.environ, {"MY_TEST_SECRET": "env_value"}):
            assert SecretManager.get_secret("MY_TEST_SECRET", tmp_path) == "env_value"
    finally:
        os.remove(tmp_path)


def test_is_kubernetes() -> None:
    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
        assert is_kubernetes() is True


def test_hallucination_detector() -> None:
    detector = HallucinationDetector()
    # Length below min (5)
    res = detector.evaluate("no")
    assert res.passed is True  # score = 0.4 < 0.5
    assert "response too short" in res.reasons

    # Suspicious phrase
    res = detector.evaluate("This is 100% accurate guaranteed always works.")
    assert res.passed is False  # score = 0.2 * 3 = 0.6 >= 0.5

    # Placeholder content
    res = detector.evaluate("Implement it soon. TODO: check this.")
    assert res.passed is True  # score = 0.3 < 0.5


def test_response_validator() -> None:
    validator = ResponseValidator()
    validator.validate("perfect response")  # should not raise
    with pytest.raises(ResponseValidationError):
        validator.validate("")
    with pytest.raises(ResponseValidationError):
        validator.validate("abc")


# =============================================================================
# 5. Telemetry & Monitoring Tests
# =============================================================================
def test_model_quality_metrics() -> None:
    metrics = ModelQualityMetrics("openai", "gpt-4")
    metrics.requests += 1
    metrics.failures += 1
    metrics.total_latency_seconds += 2.5
    metrics.hallucination_failures += 1

    # Basic state check
    assert metrics.requests == 1
    assert metrics.failures == 1
    assert metrics.total_latency_seconds == 2.5
    assert metrics.hallucination_failures == 1


# =============================================================================
# 6. CLI Argument Parsing & Signatures Tests
# =============================================================================
def test_cli_parser() -> None:
    parser = build_parser()
    # Simple parse
    args = parser.parse_args(["-p", "openai", "-q", "hello", "--stream"])
    assert args.provider == "openai"
    assert args.prompt == "hello"
    assert args.stream is True


def test_cli_build_ask_kwargs() -> None:
    # Ensure build_ask_kwargs produces expected dictionary keys matching signature of ask()
    kwargs = _build_ask_kwargs("openai", "hello", "gpt-4", 30, stream=True)
    assert kwargs["provider"] == "openai"
    assert kwargs["prompt"] == "hello"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["timeout"] == 30
    # ask doesn't currently accept 'stream' or 'profile' in core/api.py so they should be filtered out
    assert "stream" not in kwargs
    assert "profile" not in kwargs


def test_cli_decode_chunk() -> None:
    assert _decode_chunk(b"abc") == "abc"
    assert _decode_chunk("xyz") == "xyz"
    assert _decode_chunk({"result": 42}) == '{"result": 42}'


# =============================================================================
# 7. Additional resilience.py checks
# =============================================================================
def test_retry_engine_rejects_coroutine() -> None:
    engine = RetryEngine()
    async def async_fn():
        pass
    with pytest.raises(TypeError) as exc:
        engine.execute(async_fn)
    assert "Use AsyncRetryEngine" in str(exc.value)


@pytest.mark.asyncio
async def test_async_circuit_breaker() -> None:
    cb = CircuitBreaker(threshold=2, timeout=1)

    # Wrap an async function
    @cb.wrap
    async def async_fail():
        raise ValueError("async fail")

    with pytest.raises(ValueError):
        await async_fail()

    assert cb.allow() is True

    with pytest.raises(ValueError):
        await async_fail()

    assert cb.allow() is False


@pytest.mark.asyncio
async def test_async_rate_limiter() -> None:
    limiter = RateLimiter(capacity=1, rate_per_second=10)

    @limiter.wrap
    async def async_fn():
        return "ok"

    assert await async_fn() == "ok"

    with pytest.raises(RuntimeError):
        await async_fn()


def test_stream_consumer_exception() -> None:
    class BadProvider:
        def send(self, prompt):
            raise RuntimeError("provider fail")

    consumer = StreamConsumer(BadProvider())
    tokens = []
    def on_token(t):
        if t:
            tokens.append(t)

    consumer.stream("hello", on_token)
    time.sleep(0.1)
    # The consumer should output the error message in the stream and signal the sentinel
    assert len(tokens) == 1
    assert "[ERROR: provider fail]" in tokens[0]
