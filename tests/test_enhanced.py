from ai_cli.rag.chunker import chunk_text
from ai_cli.rag.embeddings import EmbeddingGenerator


def test_chunk_text_basic():
    text = "hello world this is a test"
    chunks = chunk_text(text, source="unit-test")

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(hasattr(c, "text") for c in chunks)
    assert all(hasattr(c, "source") for c in chunks)
    assert chunks[0].source == "unit-test"


def test_chunk_overlap_behavior():
    text = "a " * 100
    chunks = chunk_text(text, source="test")

    assert len(chunks) >= 1


def test_embedding_generator_smoke(_monkeypatch):
    """
    Avoid real SentenceTransformer load.
    """

    try:
        import importlib
        np = importlib.import_module("numpy")
    except ImportError:
        import unittest
        raise unittest.SkipTest("numpy is not available")

    class FakeModel:
        def encode(self, texts, convert_to_numpy=True):
            # reference convert_to_numpy to satisfy linters and emulate behavior
            if convert_to_numpy:
                return np.ones((len(texts), 10), dtype=float)
            else:
                # return a Python list of lists when conversion is not requested
                return [[1.0] * 10 for _ in texts]

    emb = EmbeddingGenerator.__new__(EmbeddingGenerator)
    emb.model = FakeModel()
    emb.normalize = True
    emb.batch_size = 2

    result = emb.embed_text("hello")

    assert result.shape == (10,)
    assert isinstance(result, np.ndarray)