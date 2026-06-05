import importlib

try:
    pytest = importlib.import_module("pytest")
except Exception:
    class _PytestShim:
        @staticmethod
        def importorskip(name):
            try:
                return importlib.import_module(name)
            except ImportError:
                raise RuntimeError(f"Required module '{name}' not available")
    pytest = _PytestShim()

np = pytest.importorskip("numpy")

from ai_cli.rag.chunker import chunk_text
from ai_cli.rag.embeddings import EmbeddingGenerator


def test_chunk_text_basic():
    text = "hello world this is a test"
    chunks = chunk_text(text, source="unit-test")
    assert isinstance(chunks, list) and chunks  # non-empty list
    assert all(hasattr(c, "text") and hasattr(c, "source") for c in chunks)
    assert chunks[0].source == "unit-test"


def test_chunk_overlap_behavior():
    text = " ".join(["a"] * 100)
    chunks = chunk_text(text, source="test")
    assert len(chunks) >= 1


def test_embedding_generator_smoke():
    # avoid loading real SentenceTransformer; require numpy via pytest.importorskip above
    class FakeModel:
        def encode(self, texts, convert_to_numpy=True):
            if convert_to_numpy:
                return np.ones((len(texts), 10), dtype=float)
            return [[1.0] * 10 for _ in texts]

    # bypass real constructor to avoid heavy initialization
    emb = EmbeddingGenerator.__new__(EmbeddingGenerator)
    emb.model = FakeModel()
    emb.normalize = True
    emb.batch_size = 2

    result = emb.embed_text("hello")
    assert isinstance(result, np.ndarray)
    assert result.shape == (10,)
