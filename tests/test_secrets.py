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