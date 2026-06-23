from ai_cli.utils import secrets


def test_secret_chunk_text():

    result = secrets.chunk_text(
        "hello world",
        chunk_size=5
    )

    assert result