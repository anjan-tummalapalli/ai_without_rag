class FakeClient:
    def __init__(self, response):
        self.response = response

    def send(self, *args, **kwargs):
        return self.response
    
def test_gemini_provider_basic():
    from ai_cli.providers.gemini_provider import GeminiProvider
    provider = GeminiProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None

def test_cohere_provider_basic():
    from ai_cli.providers.cohere_provider import CohereProvider
    provider = CohereProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None

def test_xai_provider():
    from ai_cli.providers.xAI_provider import XAIProvider
    provider = XAIProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None

def test_zai_provider():
    from ai_cli.providers.zAI_provider import ZAIProvider
    provider = ZAIProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None

def test_deepseek_provider():
    from ai_cli.providers.deepseek_provider import DeepSeekProvider
    provider = DeepSeekProvider(api_key="test")
    result = provider.send("hello")
    assert result is not None