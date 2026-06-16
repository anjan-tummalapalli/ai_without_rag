import importlib


def test_xai_provider_all():
    # Dynamically import the module to access its __all__
    module = importlib.import_module('ai_cli.providers.xAI_provider')
    assert hasattr(module, '__all__'), "Module should define __all__"
    assert 'XAIProvider' in module.__all__, "XAIProvider should be exported"
    assert 'InMemoryVectorStore' in module.__all__, "InMemoryVectorStore should be exported"