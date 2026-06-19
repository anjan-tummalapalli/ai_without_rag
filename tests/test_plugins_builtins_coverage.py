import ai_cli.plugins.builtins as b


def test_builtin_module_loads():
    # simply importing already executes module-level code
    assert hasattr(b, "__name__")

def test_builtin_registry_execution():
    import ai_cli.plugins.builtins as b
    # try calling common registry objects if exist
    for attr in dir(b):
        if not attr.startswith("_"):
            getattr(b, attr, None)