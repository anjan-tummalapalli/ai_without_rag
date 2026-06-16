from __future__ import annotations

from unittest.mock import MagicMock, patch

# pytest is not imported directly because this test module relies on pytest's
# fixture injection at runtime; remove the unused import to avoid editor/linter
# errors when pytest isn't installed in the environment.
from ai_cli.providers.base import EchoProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider


# Helpers to reduce repetition
def _setup_openai_mock(openai_mock, chat_text="response", embedding_vec=None):
    client = MagicMock()
    openai_mock.return_value = client

    # chat response
    choice = MagicMock()
    choice.message.content = chat_text
    resp = MagicMock()
    resp.choices = [choice]
    client.chat.completions.create.return_value = resp

    # embeddings
    emb_item = MagicMock()
    emb_item.embedding = embedding_vec if embedding_vec is not None else [0.1, 0.2]
    emb_resp = MagicMock()
    emb_resp.data = [emb_item]
    client.embeddings.create.return_value = emb_resp

    return client


def _setup_genai_mock(genai_mock, text="gemini response", embedding_vec=None):
    model = MagicMock()
    genai_mock.GenerativeModel.return_value = model
    resp = MagicMock()
    resp.text = text
    model.generate_content.return_value = resp

    emb_item = MagicMock()
    emb_item.embedding = embedding_vec if embedding_vec is not None else [0.1, 0.2]
    emb_resp = MagicMock()
    emb_resp.data = [emb_item]
    genai_mock.embeddings.create.return_value = emb_resp

    return model


# EchoProvider
def test_echo_provider():
    p = EchoProvider()
    assert p.provider_name == "echo"
    assert p.send("hello") == "(echo) hello"


# OpenAIProvider tests (send + RAG)
@patch("ai_cli.providers.openai_provider.OpenAI")
def test_openai_provider_send(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="openai response")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    p = OpenAIProvider()
    assert p.send("hello") == "openai response"
    openai_mock.return_value.chat.completions.create.assert_called_once()


@patch("ai_cli.providers.openai_provider.OpenAI")
def test_openai_provider_rag(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="rag response", embedding_vec=[0.1, 0.2, 0.3])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    p = OpenAIProvider()
    chunks = p.chunk_text("hello world this is a test document", chunk_size=2, overlap=1)
    assert chunks

    p.build_vector_store([{"id": "doc1", "text": "hello world"}])
    assert p._vectors is not None

    assert p.answer_with_rag("hello query") == "rag response"


# GeminiProvider
@patch("ai_cli.providers.gemini_provider.genai")
def test_gemini_provider(genai_mock, monkeypatch):
    _setup_genai_mock(genai_mock, text="gemini response")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    p = GeminiProvider(embedding_model="models/embedding-001")
    assert p.send("hello") == "gemini response"

    p.index_document("doc1", "gemini chunk text")
    assert p.send_with_rag("hello query") == "gemini response"


# CohereProvider
@patch("cohere.Client")
def test_cohere_provider(cohere_client_cls, monkeypatch):
    client = MagicMock()
    cohere_client_cls.return_value = client

    chat_resp = MagicMock()
    chat_resp.text = "cohere response"
    client.chat.return_value = chat_resp

    embed_resp = MagicMock()
    embed_resp.embeddings = [[0.1, 0.2]]
    client.embed.return_value = embed_resp

    monkeypatch.setenv("COHERE_API_KEY", "test-key")

    p = CohereProvider(rag_enabled=True)
    assert p.send("hello") == "cohere response"

    p.upsert_documents(["cohere document"])
    results = p.query_documents("query")
    assert results


# DeepSeekProvider
@patch("ai_cli.providers.deepseek_provider.OpenAI")
def test_deepseek_provider(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="deepseek response", embedding_vec=[0.1, 0.2])
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    p = DeepSeekProvider()
    assert p.ask("hello") == "deepseek response"

    embs = p.embeddings(["hello"])
    assert len(embs) == 1 and embs[0] == [0.1, 0.2]


# PerplexityProvider
@patch("ai_cli.providers.perplexity_provider.OpenAI")
def test_perplexity_provider(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="perplexity response", embedding_vec=[0.1, 0.2])
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    p = PerplexityProvider()
    assert p.send("hello") == "perplexity response"

    p.build_rag_index(["perplexity document"])
    ans, hits = p.query_with_rag("query")
    assert ans == "perplexity response"


# XAIProvider
@patch("ai_cli.providers.xAI_provider.OpenAI")
def test_xai_provider(openai_mock, monkeypatch):
    _setup_openai_mock(openai_mock, chat_text="xai response", embedding_vec=[0.1, 0.2])
    monkeypatch.setenv("XAI_API_KEY", "test-key")

    p = XAIProvider()
    assert p.send("hello") == "xai response"

    p.add_documents(["xai document"])
    assert p.send_rag("query") == "xai response"
