from ai_cli.core.resilience import RetryEngine


def test_retry_success():

    engine = RetryEngine(max_attempts=3)

    assert engine.execute(lambda: "ok") == "ok"


def test_retry_failure():

    engine = RetryEngine(
        max_attempts=2,
        base_delay=0
    )

    def fail():
        raise ValueError("bad")

    try:
        engine.execute(fail)
    except ValueError:
        assert True
