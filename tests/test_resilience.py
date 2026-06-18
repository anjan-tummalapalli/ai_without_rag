from ai_cli.core.resilience import execute_with_fallback


def test_resilience_fallback():
    def fail():
        raise ValueError("fail")

    def fallback():
        return "ok"
    assert execute_with_fallback(fail, fallback) == "ok"