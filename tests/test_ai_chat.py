# ai_cli/rag.py
from __future__ import annotations

import asyncio
import hashlib
import heapq
import math
from collections.abc import Iterable
from unittest.mock import patch

import pytest

import ai_cli.cli as cli
from ai_cli.ai_chat import (
    _last_sentence_end,
    _last_whitespace,
    _next_start,
    chunk_text,
)

print("### LOADED test_ai_chat.py ###")


def test_last_sentence_end_found():
    text = "Hello world. Next"
    assert _last_sentence_end(text) == 11


def test_last_sentence_end_missing():
    text = "Hello world"
    assert _last_sentence_end(text) == -1


def test_last_whitespace_found():
    text = "hello world"
    assert _last_whitespace(text) == 5


def test_last_whitespace_missing():
    text = "helloworld"
    assert _last_whitespace(text) == -1


def test_next_start():
    assert _next_start(end=10, chunk_overlap=3, prev_start=0) == 7


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_basic():
    result = chunk_text(
        "This is a simple sentence. Another sentence here.",
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(result) > 0
    assert isinstance(result[0], str)


def test_ai_chat_chunking():
    result = chunk_text(
        "hello world this is a test",
        chunk_size=10,
        chunk_overlap=2,
    )
    assert isinstance(result, list)
    assert len(result) > 0


def test_ai_chat_empty():
    result = chunk_text(
        "",
        chunk_size=10,
        chunk_overlap=2,
    )
    assert result == []


# Deterministic "embedding" using hash -> fixed-dim float vector in [-1,1]
def _text_to_embedding(text: str, dim: int = 64) -> list[float]:
    if not isinstance(text, str):
        text = ""
    h = hashlib.sha256(text.encode("utf8")).digest()
    vec: list[float] = []
    i = 0
    # iterate over hash bytes, repeat as needed
    while len(vec) < dim:
        b = h[i % len(h)]
        vec.append((b / 255.0) * 2.0 - 1.0)
        i += 1
    return vec[:dim]


def embed_texts(texts: Iterable[str], dim: int = 64) -> list[list[float]]:
    return [_text_to_embedding(t or "", dim=dim) for t in texts]


# Cosine similarity helpers
def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _norm(a: list[float]) -> float:
    s = sum(x * x for x in a)
    return math.sqrt(s) if s > 0.0 else 0.0


# Simple in-memory vector DB with small optimizations
class VectorStore:
    def __init__(self, dim: int = 64):
        self.dim = dim
        self._docs: dict[str, str] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._norms: dict[str, float] = {}

    def add(self, doc_id: str, text: str) -> None:
        emb = _text_to_embedding(text, dim=self.dim)
        self._docs[doc_id] = text
        self._embeddings[doc_id] = emb
        self._norms[doc_id] = _norm(emb)

    def add_many(self, items: Iterable[tuple[str, str]]) -> None:
        for doc_id, text in items:
            self.add(doc_id, text)

    def query(
        self, query_text: str, top_k: int = 3, min_score: float = 0.0
    ) -> list[tuple[str, str, float]]:
        """
        Returns top_k (doc_id, text, score) ordered by score desc.
        Uses a min-heap of size top_k for O(n log k) selection.
        """
        q_emb = _text_to_embedding(query_text, dim=self.dim)
        q_norm = _norm(q_emb)
        if q_norm == 0.0:
            return []

        heap: list[tuple[float, str, str]] = []  # (score, doc_id, text)
        for doc_id, emb in self._embeddings.items():
            doc_norm = self._norms.get(doc_id, 0.0)
            if doc_norm == 0.0:
                continue
            score = _dot(q_emb, emb) / (q_norm * doc_norm)
            if score < min_score:
                continue
            if len(heap) < top_k:
                heapq.heappush(heap, (score, doc_id, self._docs[doc_id]))
            else:
                # heap[0] is smallest score in heap, replace if current score is higher
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, doc_id, self._docs[doc_id]))

        # convert heap to sorted list (desc)
        results = [(doc_id, text, score) for score, doc_id, text in heap]
        results.sort(key=lambda t: t[2], reverse=True)
        return results

    def all_docs(self) -> dict[str, str]:
        return dict(self._docs)


# Convenience: build a VectorStore from a long document (chunked)
def build_store_from_text(
    doc_id_prefix: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    dim: int = 64,
) -> VectorStore:
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    store = VectorStore(dim=dim)
    for i, c in enumerate(chunks):
        store.add(f"{doc_id_prefix}-{i}", c)
    return store


def test_last_whitespace_tabs():
    # Covers line 42-43:
    # for match in _WHITESPACE_RE.finditer(window)
    result = _last_whitespace("hello\tworld")

    assert result == 5


def test_chunk_text_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text(
            "hello world",
            chunk_size=0,
        )


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "hello world",
            chunk_size=5,
            chunk_overlap=5,
        )


def test_chunk_text_whitespace_only():
    assert chunk_text("     ") == []


def test_chunk_text_sentence_boundary():
    result = chunk_text(
        "Hello world. Next sentence continues here",
        chunk_size=20,
        chunk_overlap=5,
        prefer_sentence_boundary=True,
    )

    assert len(result) >= 1


def test_chunk_text_without_word_split():
    result = chunk_text(
        "abcdefghij klmnop",
        chunk_size=10,
        chunk_overlap=2,
        split_on_word=False,
    )

    assert len(result) >= 1


def test_chunk_text_word_split():
    result = chunk_text(
        "abcdefghij klmnop",
        chunk_size=10,
        chunk_overlap=2,
        split_on_word=True,
    )

    assert len(result) >= 1


def test_chunk_text_exact_length_break():

    text = "hello"

    result = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert result == ["hello"]


def test_chunk_text_multiple_chunks_progress():

    text = "one two three four five six seven eight nine"

    result = chunk_text(
        text,
        chunk_size=10,
        chunk_overlap=2,
        split_on_word=False,
    )

    assert len(result) > 1


def test_chunk_text_empty_after_strip_exit():
    result = chunk_text(
        None,
        chunk_size=10,
        chunk_overlap=2,
    )

    assert result == []


def test_chunk_text_empty_chunk_after_strip():

    result = chunk_text(
        "   ",
        chunk_size=5,
        chunk_overlap=1,
    )

    assert result == []


def test_build_kwargs_modules():
    kwargs = cli._build_ask_kwargs(
        provider="auto",
        prompt="hi",
        model=None,
        timeout=10,
        modules=" aws , kubernetes , python ,, ",
    )

    assert kwargs["modules"] == [
        "aws",
        "kubernetes",
        "python",
    ]


@patch("ai_cli.cli.inspect.signature")
def test_build_kwargs_signature_failure(mock_sig):
    mock_sig.side_effect = TypeError

    kwargs = cli._build_ask_kwargs(
        provider="openai",
        prompt="hello",
        model="gpt",
        timeout=20,
        profile="prod",
        stream=True,
        modules="aws,k8s",
    )

    assert kwargs["provider"] == "openai"
    assert kwargs["model"] == "gpt"
    assert kwargs["profile"] == "prod"
    assert kwargs["stream"] is True
    assert kwargs["modules"] == ["aws", "k8s"]


@patch("ai_cli.cli.json.dumps")
def test_decode_chunk_json_failure(mock_dump):
    mock_dump.side_effect = TypeError

    class Bad:
        def __str__(self):
            return "BAD"

    assert cli._decode_chunk(Bad()) == "BAD"


class AsyncGen:
    def __aiter__(self):
        return self

    async def __anext__(self):
        if hasattr(self, "done"):
            raise StopAsyncIteration
        self.done = True
        return "hello"


async def boom():
    raise KeyboardInterrupt()


assert asyncio.run(cli._drain_async_result(boom())) == 130


async def coro():
    return "hello"


assert cli._handle_sync_result(coro()) == 0


@patch("ai_cli.cli._decode_chunk")
def test_handle_sync_runtime_error(mock_decode):
    mock_decode.side_effect = RuntimeError("boom")

    assert cli._handle_sync_result("abc") == 1
