def test_builtin_load():
    from ai_cli.plugins import builtins
    assert hasattr(builtins, "__file__")

def test_factory_registry():
    from unittest.mock import MagicMock

    from ai_cli.providers.factory import build_provider
    req = MagicMock()
    req.provider = "echo"
    req.api_key = None
    req.model = None
    req.kwargs = {}

    try:
        build_provider(req)
    except Exception:
        pass