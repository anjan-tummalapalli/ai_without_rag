import pytest

from types import SimpleNamespace

from ai_cli.providers import gemini_provider
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.gemini_provider import _GenaiShim
from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.gemini_provider import GeminiProvider

def test_shim_configure():
    shim = _GenaiShim()

    with pytest.raises(ProviderRequestError):
        shim.configure()


def test_shim_model():
    with pytest.raises(ProviderRequestError):
        _GenaiShim.GenerativeModel()


def test_shim_client():
    with pytest.raises(ProviderRequestError):
        _GenaiShim.Client()


def test_shim_generate():
    with pytest.raises(ProviderRequestError):
        _GenaiShim.Client.models.generate_content()


def test_create_embeddings_empty_response(monkeypatch):
    provider = GeminiProvider(api_key="test")
    provider._use_new_api = True

    monkeypatch.setattr(
        provider,
        "_embed_with_new_sdk",
        lambda *_: [],
    )

    import pytest

    with pytest.raises(
        ProviderRequestError,
        match="Embedding API returned no data",
    ):
        provider._create_embeddings(["hello"])




class FakeModels:
    def embed_content(self, **kwargs):
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[1.0, 2.0])
            ]
        )

def test_embed_with_new_sdk_success():
    provider = GeminiProvider(api_key="test")
    provider.client = SimpleNamespace(models=FakeModels())

    result = provider._embed_with_new_sdk(
        "model",
        ["hello"],
    )

    assert result == [[1.0, 2.0]]


def test_embed_with_legacy_sdk(monkeypatch):
    provider = GeminiProvider(api_key="test")

    monkeypatch.setattr(
        gemini_provider,
        "genai",
        type(
            "FakeGenAI",
            (),
            {
                "embed_content": staticmethod(
                    lambda **_: {"embedding": [1.0, 2.0]}
                )
            },
        )(),
    )

    result = provider._embed_with_legacy_sdk(
        "model",
        ["hello"],
    )

    assert result == [[1.0, 2.0]]

def test_create_embeddings_success(monkeypatch):
    provider = GeminiProvider(api_key="test")
    provider._use_new_api = True
    monkeypatch.setattr(
        provider,
        "_embed_with_new_sdk",
        lambda *_: [[0.1, 0.2]],
    )

    result = provider._create_embeddings(["hello"])
    assert result == [[0.1, 0.2]]