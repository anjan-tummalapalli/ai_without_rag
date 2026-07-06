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


def test_chunk_text_invalid_chunk_size_zero():
    with pytest.raises(ValueError, match="chunk_size must be a positive int"):
        chunk_text("hello", chunk_size=0)


def test_chunk_text_invalid_chunk_size_type():
    with pytest.raises(ValueError, match="chunk_size must be a positive int"):
        chunk_text("hello", chunk_size="10")  # type: ignore[arg-type]


def test_chunk_text_invalid_overlap_negative():
    with pytest.raises(ValueError, match="overlap must be a non-negative int"):
        chunk_text("hello", overlap=-1)


def test_chunk_text_invalid_overlap_type():
    with pytest.raises(ValueError, match="overlap must be a non-negative int"):
        chunk_text("hello", overlap="5")  # type: ignore[arg-type]


def test_chunk_text_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError, match="greater than overlap"):
        chunk_text("hello world", chunk_size=5, overlap=6)


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("   \n\t   ") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "short text"
    assert chunk_text(text, chunk_size=100) == [text]