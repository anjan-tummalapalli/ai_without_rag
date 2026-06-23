import builtins

from ai_cli import cli
from ai_cli.rag.pipeline import RAGPipeline


def test_interactive_exit(monkeypatch):
    inputs = iter(["/exit"])

    monkeypatch.setattr(
        builtins,
        "input",
        lambda _: next(inputs),
    )

    result = cli.run_interactive(
        provider="mock",
        model=None,
        timeout=30,
    )

    assert result == 0


def test_rag_retrieve_empty():
    rag = RAGPipeline()

    assert rag.retrieve_context("hello") == ""


def test_rag_upsert_and_search():
    rag = RAGPipeline()

    rag.upsert_documents(
        ["hello world document"]
    )

    result = rag.retrieve_context("hello")

    assert "hello" in result
