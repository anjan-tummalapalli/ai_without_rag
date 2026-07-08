from ai_cli.ai_chat import chunk_text


def test_chunk_text_basic():
    result = chunk_text(
        "This is a test sentence. Another sentence.",
        chunk_size=20,
        chunk_overlap=5,
    )

    assert result
    assert isinstance(result, list)


def test_chunk_text_empty():
    result = chunk_text(
        "",
        chunk_size=20,
        chunk_overlap=5,
    )

    assert result == []


def test_chunk_text_overlap():
    result = chunk_text(
        "one two three four five six seven",
        chunk_size=10,
        chunk_overlap=2,
    )

    assert len(result) > 1
