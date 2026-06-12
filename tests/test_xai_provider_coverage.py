from unittest.mock import patch, MagicMock
from ai_cli.providers import __all__ as provider_all


def _setup_openai_mock(openai_mock, chat_text="response", embedding_vec=None):
    client = MagicMock()
    openai_mock.return_value = client

    # chat response
    choice = MagicMock()
    choice.message.content = chat_text
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp

    # embeddings response
    emb_item = MagicMock()
    emb_item.embedding = embedding_vec if embedding_vec is not None else [0.1, 0.2]
    emb_resp = MagicMock()
    emb_resp.data = [emb_item]
    client.embeddings.create.return_value = emb_resp
    return client

@patch("ai_cli.providers.xAI_provider.OpenAI")
def test_xai_provider_send(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="xai hello")
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    from ai_cli.providers.xAI_provider import XAIProvider
    prov = XAIProvider()
    assert prov.send("hi") == "xai hello"
    openai_mock.return_value.chat.completions.create.assert_called_once()

@patch("ai_cli.providers.xAI_provider.OpenAI")
def test_xai_provider_send_rag_default_prompt(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="rag answer", embedding_vec=[0.1, 0.2])
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    from ai_cli.providers.xAI_provider import XAIProvider
    prov = XAIProvider()
    prov.add_documents(["sample document text for testing"])
    result = prov.send_rag("what is this?")
    assert result == "rag answer"

def test_provider_all_exports():
    assert "XAIProvider" in provider_all
    assert "InMemoryVectorStore" in provider_all
