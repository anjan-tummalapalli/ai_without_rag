from unittest.mock import patch

import ai_cli.cli as cli
from ai_cli.telemetry.monitoring import Monitoring


def test_interactive_keyboard_interrupt():
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        assert (
            cli.run_interactive(
                provider="openai",
                model=None,
                profile=None,
                stream=False,
                timeout=30,
                modules=None,
                rag=None,
                rag_chunk_size=100,
                rag_chunk_overlap=10,
                rag_top_k=3,
            )
            == 0
        )


def test_empty_input_then_exit():
    with patch(
        "builtins.input",
        side_effect=["", "/quit"],
    ):
        assert (
            cli.run_interactive(
                provider="openai",
                model=None,
                profile=None,
                stream=False,
                timeout=30,
                modules=None,
                rag=None,
                rag_chunk_size=100,
                rag_chunk_overlap=10,
                rag_top_k=3,
            )
            == 0
        )


def test_switch_without_provider():
    with patch(
        "builtins.input",
        side_effect=["/switch", "/quit"],
    ):
        cli.run_interactive(
            provider="openai",
            model=None,
            profile=None,
            stream=False,
            timeout=30,
            modules=None,
            rag=None,
            rag_chunk_size=100,
            rag_chunk_overlap=10,
            rag_top_k=3,
        )


def test_model_clear():
    with patch(
        "builtins.input",
        side_effect=["/model", "/quit"],
    ):
        cli.run_interactive(
            provider="openai",
            model="abc",
            profile=None,
            stream=False,
            timeout=30,
            modules=None,
            rag=None,
            rag_chunk_size=100,
            rag_chunk_overlap=10,
            rag_top_k=3,
        )


def test_index_usage():
    with patch(
        "builtins.input",
        side_effect=["/index", "/quit"],
    ):
        cli.run_interactive(
            provider="openai",
            model=None,
            profile=None,
            stream=False,
            timeout=30,
            modules=None,
            rag=None,
            rag_chunk_size=100,
            rag_chunk_overlap=10,
            rag_top_k=3,
        )


def test_large_prompt(monkeypatch):
    huge = "A" * 150000

    monkeypatch.setattr(
        cli,
        "_invoke_with_retries",
        lambda *a, **k: 0,
    )

    class FakePipeline:
        def retrieve_context(self, *a, **k):
            return ""

    with patch(
        "builtins.input",
        side_effect=[huge, "/quit"],
    ):
        cli.run_interactive(
            provider="openai",
            model=None,
            profile=None,
            stream=False,
            timeout=30,
            modules=None,
            rag=FakePipeline(),
            rag_chunk_size=100,
            rag_chunk_overlap=10,
            rag_top_k=3,
        )


class BadMetric:
    def labels(self, **kwargs):
        raise RuntimeError()


def test_record_request_exception():
    m = Monitoring()
    m.requests = BadMetric()

    m.record_request("openai")


def test_record_failure_exception():
    m = Monitoring()
    m.failures = BadMetric()

    m.record_failure("openai")


def test_record_latency_exception():
    m = Monitoring()
    m.latency = BadMetric()

    m.record_latency("openai", 2.5)


def test_record_chunks_exception():
    m = Monitoring()
    m.chunks = BadMetric()

    m.record_chunks("openai", "model", 5)


def test_record_embedding_exception():
    m = Monitoring()
    m.embedding_requests = BadMetric()

    m.record_embedding("openai", "model", 1.2)


def test_record_vector_query_exception():
    m = Monitoring()
    m.vector_queries = BadMetric()

    m.record_vector_query("openai", "model")
