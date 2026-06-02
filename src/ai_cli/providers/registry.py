# ai_cli/providers/registry.py
# Provide robust imports: try relative imports first (normal when package is installed),
# then try absolute imports (useful in some tooling/environments), and finally
# expose lightweight placeholders that raise clear runtime errors if the real
# implementations are not available so static analyzers / importers don't report errors.
import importlib
from typing import Any, Callable

def _missing_callable(name: str) -> Callable[..., Any]:
    def _fn(*_, **__):
        raise RuntimeError(
            f"The optional module 'ai_cli.providers.rag.{name}' is not available; "
            "please ensure the 'rag' subpackage is installed or available on PYTHONPATH."
        )
    return _fn

# Prepare defaults that will be overwritten if imports succeed
chunk_text = None
chunk_documents = None
EmbeddingModel = None
LocalHashEmbedding = None
OpenAIEmbedding = None
InMemoryVectorDB = None
build_index_from_texts = None

# Candidate package bases to try: prefer relative to this package, then absolute
_candidates = []
if __package__:
    _candidates.append(f"{__package__}.rag")
_candidates.append("ai_cli.providers.rag")

def _import_from(candidate_base: str, submodule: str, names):
    try:
        mod = importlib.import_module(f"{candidate_base}.{submodule}")
    except Exception:
        return None
    result = {}
    for n in names:
        result[n] = getattr(mod, n)
    return result

for base in _candidates:
    chunker = _import_from(base, "chunker", ["chunk_text", "chunk_documents"])
    embeddings = _import_from(base, "embeddings", ["EmbeddingModel", "LocalHashEmbedding", "OpenAIEmbedding"])
    vector_db = _import_from(base, "vector_db", ["InMemoryVectorDB", "build_index_from_texts"])
    if chunker and embeddings and vector_db:
        chunk_text = chunker["chunk_text"]
        chunk_documents = chunker["chunk_documents"]
        EmbeddingModel = embeddings["EmbeddingModel"]
        LocalHashEmbedding = embeddings["LocalHashEmbedding"]
        OpenAIEmbedding = embeddings["OpenAIEmbedding"]
        InMemoryVectorDB = vector_db["InMemoryVectorDB"]
        build_index_from_texts = vector_db["build_index_from_texts"]
        break

# If any required symbol is still missing, provide clear runtime placeholders.
if chunk_text is None:
    chunk_text = _missing_callable("chunker.chunk_text")
if chunk_documents is None:
    chunk_documents = _missing_callable("chunker.chunk_documents")

if EmbeddingModel is None:
    class EmbeddingModel:
        def __init__(self, *_, **__):
            raise RuntimeError(
                "EmbeddingModel requires 'ai_cli.providers.rag.embeddings' to be available."
            )

    class LocalHashEmbedding(EmbeddingModel):
        pass

    class OpenAIEmbedding(EmbeddingModel):
        pass

if InMemoryVectorDB is None:
    class InMemoryVectorDB:
        def __init__(self, *_, **__):
            raise RuntimeError(
                "InMemoryVectorDB requires 'ai_cli.providers.rag.vector_db' to be available."
            )

if build_index_from_texts is None:
    def build_index_from_texts(*_, **__):
        raise RuntimeError(
            "build_index_from_texts requires 'ai_cli.providers.rag.vector_db' to be available."
        )

__all__ = [
    "chunk_text",
    "chunk_documents",
    "EmbeddingModel",
    "LocalHashEmbedding",
    "OpenAIEmbedding",
    "InMemoryVectorDB",
    "build_index_from_texts",
]
