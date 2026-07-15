import builtins

from ai_cli.providers.loader import load_all_providers


def test_loader_without_gemini(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "ai_cli.providers.gemini_provider":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    providers = load_all_providers()

    assert "gemini" not in providers
