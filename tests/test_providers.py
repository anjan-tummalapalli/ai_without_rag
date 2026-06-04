from __future__ import annotations

import os
from unittest.mock import patch, MagicMock
import pytest

from ai_cli.core.exceptions import ProviderRequestError
from ai_cli.providers.base import EchoProvider
from ai_cli.providers.openai_provider import OpenAIProvider
from ai_cli.providers.gemini_provider import GeminiProvider
from ai_cli.providers.cohere_provider import CohereProvider
from ai_cli.providers.deepseek_provider import DeepSeekProvider
from ai_cli.providers.perplexity_provider import PerplexityProvider
from ai_cli.providers.xAI_provider import XAIProvider


# ---------------------------------------------------------
# EchoProvider Tests
# ---------------------------------------------------------
def test_echo_provider():
    provider = EchoProvider()
    assert provider.provider_name == "echo"
    assert provider.send("hello") == "(echo) hello"


# ---------------------------------------------------------
# OpenAIProvider Tests
# ---------------------------------------------------------
@patch("ai_cli.providers.openai_provider.OpenAI")
def test_openai_provider_send(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock chat completion response
    mock_choice = MagicMock()
    mock_choice.message.content = "openai response"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider()
        assert provider.send("hello") == "openai response"
        mock_client.chat.completions.create.assert_called_once()


@patch("ai_cli.providers.openai_provider.OpenAI")
def test_openai_provider_rag(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client

    # Mock embeddings response
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3]
    mock_emb_resp = MagicMock()
    mock_emb_resp.data = [mock_embedding]
    mock_client.embeddings.create.return_value = mock_emb_resp

    # Mock chat response
    mock_choice = MagicMock()
    mock_choice.message.content = "rag response"
    mock_chat_resp = MagicMock()
    mock_chat_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_chat_resp

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        provider = OpenAIProvider()
        # chunk text
        chunks = provider.chunk_text("hello world this is a test document", chunk_size=2, overlap=1)
        assert len(chunks) > 0

        # build store
        provider.build_vector_store([{"id": "doc1", "text": "hello world"}])
        assert provider._vectors is not None

        # query
        res = provider.answer_with_rag("hello query")
        assert res == "rag response"


# ---------------------------------------------------------
# GeminiProvider Tests
# ---------------------------------------------------------
@patch("ai_cli.providers.gemini_provider.genai")
def test_gemini_provider(mock_genai):
    mock_model = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    
    mock_resp = MagicMock()
    mock_resp.text = "gemini response"
    mock_model.generate_content.return_value = mock_resp

    # Mock embeddings
    mock_emb_item = MagicMock()
    mock_emb_item.embedding = [0.1, 0.2]
    mock_emb_resp = MagicMock()
    mock_emb_resp.data = [mock_emb_item]
    mock_genai.embeddings.create.return_value = mock_emb_resp

    with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
        provider = GeminiProvider(embedding_model="models/embedding-001")
        assert provider.send("hello") == "gemini response"

        # RAG index
        provider.index_document("doc1", "gemini chunk text")
        
        # RAG query
        res = provider.send_with_rag("hello query")
        assert res == "gemini response"


# ---------------------------------------------------------
# CohereProvider Tests
# ---------------------------------------------------------
@patch("cohere.Client")
def test_cohere_provider(mock_cohere_client_cls):
    mock_client = MagicMock()
    mock_cohere_client_cls.return_value = mock_client
    
    # Mock chat response
    mock_chat_resp = MagicMock()
    mock_chat_resp.text = "cohere response"
    mock_client.chat.return_value = mock_chat_resp

    # Mock embed response
    mock_embed_resp = MagicMock()
    mock_embed_resp.embeddings = [[0.1, 0.2]]
    mock_client.embed.return_value = mock_embed_resp

    with patch.dict(os.environ, {"COHERE_API_KEY": "test-key"}):
        provider = CohereProvider(rag_enabled=True)
        assert provider.send("hello") == "cohere response"

        # Upsert docs
        provider.upsert_documents(["cohere document"])
        
        # Test query
        results = provider.query_documents("query")
        assert len(results) > 0


# ---------------------------------------------------------
# DeepSeekProvider Tests
# ---------------------------------------------------------
@patch("ai_cli.providers.deepseek_provider.OpenAI")
def test_deepseek_provider(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock chat response
    mock_choice = MagicMock()
    mock_choice.message.content = "deepseek response"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    # Mock embed response
    mock_emb_item = MagicMock()
    mock_emb_item.embedding = [0.1, 0.2]
    mock_emb_resp = MagicMock()
    mock_emb_resp.data = [mock_emb_item]
    mock_client.embeddings.create.return_value = mock_emb_resp

    with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
        provider = DeepSeekProvider()
        assert provider.ask("hello") == "deepseek response"
        
        # Embeddings
        embs = provider.embeddings(["hello"])
        assert len(embs) == 1
        assert embs[0] == [0.1, 0.2]


# ---------------------------------------------------------
# PerplexityProvider Tests
# ---------------------------------------------------------
@patch("ai_cli.providers.perplexity_provider.OpenAI")
def test_perplexity_provider(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock chat response
    mock_choice = MagicMock()
    mock_choice.message.content = "perplexity response"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    # Mock embeddings
    mock_emb_item = MagicMock()
    mock_emb_item.embedding = [0.1, 0.2]
    mock_emb_resp = MagicMock()
    mock_emb_resp.data = [mock_emb_item]
    mock_client.embeddings.create.return_value = mock_emb_resp

    with patch.dict(os.environ, {"PERPLEXITY_API_KEY": "test-key"}):
        provider = PerplexityProvider()
        assert provider.send("hello") == "perplexity response"

        # RAG index
        provider.build_rag_index(["perplexity document"])
        
        # RAG query
        ans, hits = provider.query_with_rag("query")
        assert ans == "perplexity response"


# ---------------------------------------------------------
# XAIProvider Tests
# ---------------------------------------------------------
@patch("ai_cli.providers.xAI_provider.OpenAI")
def test_xai_provider(mock_openai_class):
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock chat response
    mock_choice = MagicMock()
    mock_choice.message.content = "xai response"
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    # Mock embeddings
    mock_emb_item = MagicMock()
    mock_emb_item.embedding = [0.1, 0.2]
    mock_emb_resp = MagicMock()
    mock_emb_resp.data = [mock_emb_item]
    mock_client.embeddings.create.return_value = mock_emb_resp

    with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}):
        provider = XAIProvider()
        assert provider.send("hello") == "xai response"

        # RAG index
        provider.add_documents(["xai document"])
        
        # RAG send
        ans = provider.send_rag("query")
        assert ans == "xai response"
