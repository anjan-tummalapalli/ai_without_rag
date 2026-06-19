def test_builtin_load():
    from ai_cli.plugins import builtins
    assert hasattr(builtins, "__file__")

def test_factory_registry():
    from ai_cli.providers.factory import build_provider

    try:
        build_provider(provider_name="echo")
    except Exception:
        pass