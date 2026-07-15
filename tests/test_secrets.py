import pytest

from ai_cli.utils import secrets
from ai_cli.utils.secrets import chunk_text


def test_secret_chunk_text():
    result = secrets.chunk_text(
        "hello world",
        chunk_size=10,
    )

    assert result


def test_chunk_text_forward_progress_branch():
    text = "abcdefghijklmnopqrstuvwxyz"
    # overlap == chunk_size would normally prevent progress,
    # but implementation forces next_start=end.

    chunks = chunk_text(
        text,
        chunk_size=10,
        overlap=10,
    )

    assert len(chunks) >= 2
    assert chunks[0] == "abcdefghij"
    assert chunks[-1]


def test_chunk_text_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("hello", chunk_size=0)


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("hello", chunk_size=10, overlap=-1)


def test_chunk_text_overlap_larger_than_chunk():
    with pytest.raises(ValueError, match="greater than overlap"):
        chunk_text("hello", chunk_size=5, overlap=6)


def test_chunk_text_empty_string():
    assert chunk_text("") == []


def test_chunk_text_short_string():
    assert chunk_text("hello", chunk_size=100) == ["hello"]


def test_chunk_text_multiple_chunks():
    text = "A" * 3000

    chunks = chunk_text(
        text,
        chunk_size=1000,
        overlap=100,
    )

    assert len(chunks) >= 3
