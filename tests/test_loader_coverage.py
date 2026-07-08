# Local application imports
from ai_cli.providers import loader


def test_loader_import_exec():
    assert loader is not None
