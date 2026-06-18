from ai_cli.core.resilience import Cache


def test_cache_set_get_delete():
    cache = Cache()

    cache.set("key", "value")

    assert cache.get("key") == "value"

    cache.delete("key")

    assert cache.get("key") is None


def test_cache_clear():
    cache = Cache()

    cache.set("a", 1)
    cache.set("b", 2)

    cache.clear()

    assert cache.get("a") is None
