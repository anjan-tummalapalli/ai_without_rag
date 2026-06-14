from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import faiss  # type: ignore
    import numpy as np  # type: ignore
else:
    try:
        import faiss  # type: ignore
    except Exception:
        faiss = None  # type: ignore

    try:
        import numpy as np
    except Exception:
        np = None  # type: ignore

from ai_cli.config.rag_config import FAISS_INDEX_PATH, METADATA_PATH
from ai_cli.rag.models import Chunk


class VectorStore:
    """
    FAISS-backed vector store.
    Stores an in-memory list of Chunk objects, a matching numpy.ndarray of embeddings
    (float32) and an optional persisted FAISS index on disk.
    """
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.index = None
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self._embeddings_path = Path(str(FAISS_INDEX_PATH)).with_suffix(".npy")

    def _require_numpy(self) -> None:
        if np is None:
            raise RuntimeError("numpy is required for VectorStore operations.")

    def _require_faiss(self) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss library is not available; please install faiss to use VectorStore."
            )

    def create_index(self, dimension: int) -> None:
        self._require_faiss()
        self.index = faiss.IndexFlatL2(int(dimension))
        self.chunks = []
        self.embeddings = None
        self._embeddings_path = Path(str(FAISS_INDEX_PATH)).with_suffix(".npy")

    def _rebuild_index(self) -> None:
        """Recreate FAISS index from current embeddings (used after upsert/delete)."""
        if self.embeddings is None or self.embeddings.size == 0:
            self.index = None
            return
        dim = int(self.embeddings.shape[1])
        self.create_index(dim)
        # faiss IndexFlatL2.add expects float32 2D array
        self.index.add(self.embeddings)

    def add_embeddings(self, embeddings: Iterable[np.ndarray], chunks: list[Chunk]) -> None:
        """
        Append new vectors and chunks.
        embeddings: iterable of 1D vectors or an array-like of shape (n, dim).
        """
        self._require_numpy()
        vectors = np.array(list(embeddings)).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if self.index is None:
            self.create_index(vectors.shape[1])

        self.index.add(vectors)

        if self.embeddings is None:
            self.embeddings = vectors.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])

        self.chunks.extend(chunks)

    def upsert(self, embeddings: Iterable[np.ndarray], chunks: list[Chunk]) -> None:
        """
        Upsert: replace vectors for existing chunk ids, append new otherwise.
        Rebuilds the index if any replacement occurred.
        """
        self._require_numpy()
        vectors = np.array(list(embeddings)).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        id_to_pos = {c.id: i for i, c in enumerate(self.chunks)}

        replaced = False
        new_vectors = []
        new_chunks = []

        for vec, chunk in zip(vectors, chunks, strict=False):
            if chunk.id in id_to_pos:
                pos = id_to_pos[chunk.id]
                if self.embeddings is not None:
                    self.embeddings[pos] = vec
                self.chunks[pos] = chunk
                replaced = True
            else:
                new_vectors.append(vec)
                new_chunks.append(chunk)

        if new_vectors:
            new_np = np.array(new_vectors).astype("float32")
            if self.embeddings is None:
                self.embeddings = new_np.copy()
            else:
                self.embeddings = np.vstack([self.embeddings, new_np])
            self.chunks.extend(new_chunks)

        if replaced or self.index is None:
            self._rebuild_index()
        else:
            # only append new vectors to existing index
            if new_vectors:
                self.index.add(new_np)

    def delete(self, ids: Iterable[str]) -> None:
        """
        Delete chunks and embeddings by chunk.id and rebuild the FAISS index.
        """
        self._require_numpy()
        ids_to_remove = set(ids)
        if not ids_to_remove:
            return

        kept_embeddings = []
        kept_chunks = []
        if self.embeddings is not None:
            for emb, ch in zip(self.embeddings, self.chunks, strict=False):
                if ch.id not in ids_to_remove:
                    kept_embeddings.append(emb)
                    kept_chunks.append(ch)

        if kept_embeddings:
            self.embeddings = np.vstack(kept_embeddings).astype("float32")
        else:
            # preserve dimensionality if possible
            if self.embeddings is not None:
                d = int(self.embeddings.shape[1])
                self.embeddings = np.empty((0, d), dtype="float32")
            else:
                self.embeddings = None

        self.chunks = kept_chunks
        self._rebuild_index()

    def search(
        self,
        query_embedding,
        top_k: int = 5,
        filter_fn: Callable[[Chunk], bool] | None = None,
        return_scores: bool = True,
    ):
        """
        Similarity search with optional filtering. Returns list of {"chunk": Chunk, "score": float}.
        Score is raw L2 distance from FAISS.
        """
        if self.index is None or self.embeddings is None or len(self.chunks) == 0:
            return []

        if top_k <= 0:
            return []

        top_k = min(int(top_k), len(self.chunks))
        self._require_numpy()
        q = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(q, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if filter_fn is not None and not filter_fn(chunk):
                continue
            item = {"chunk": chunk}
            if return_scores:
                item["score"] = float(dist)
            results.append(item)

        return results

    def save(self) -> None:
        """
        Persist FAISS index (if present and faiss installed), metadata and embeddings.
        """
        # ensure metadata/embeddings dir exists
        Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(str(self._embeddings_path)).parent.mkdir(parents=True, exist_ok=True)

        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        metadata = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in self.chunks
        ]
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if self.embeddings is not None:
            self._require_numpy()
            np.save(str(self._embeddings_path), self.embeddings)

    def load(self) -> None:
        """
        Load persisted index (if faiss available), metadata and embeddings from disk.
        """
        faiss_path = Path(FAISS_INDEX_PATH)
        if faiss_path.exists() and faiss is not None:
            self.index = faiss.read_index(str(faiss_path))
        else:
            self.index = None

        if Path(METADATA_PATH).exists():
            metadata = json.loads(Path(METADATA_PATH).read_text(encoding="utf-8"))
            self.chunks = [
                Chunk(
                    id=item["id"],
                    text=item["text"],
                    source=item.get("source"),
                    chunk_index=item.get("chunk_index"),
                    metadata=item.get("metadata", {}),
                )
                for item in metadata
            ]
        else:
            self.chunks = []

        if self._embeddings_path.exists():
            self._require_numpy()
            self.embeddings = np.load(str(self._embeddings_path)).astype("float32")
        else:
            if self.index is not None and faiss is not None:
                # try to infer dimension from index
                try:
                    d = int(self.index.d)
                except Exception:
                    d = 0
                self.embeddings = np.empty((0, d), dtype="float32") if d > 0 else None
            else:
                self.embeddings = None


# Backward compatibility alias
class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store.

    Compatibility wrapper around VectorStore for tests and lightweight usage.

    Supports:
        store = InMemoryVectorStore(dim=3)
        store.add("text", [0.1, 0.2, 0.3])
        store.search([0.1, 0.2, 0.3])
    """

    def __init__(self, dim: int = 768) -> None:
        super().__init__(dim=dim)
        self.embeddings = []

    def add(self, text, embedding, metadata=None):
        """
        Add a single text/vector pair.

        Args:
            text: text content
            embedding: vector embedding
            metadata: optional metadata
        """
        self.chunks.append(
            Chunk(
                id=str(len(self.chunks)),
                text=text,
                source="memory",
                chunk_index=len(self.chunks),
                metadata=metadata or {},
            )
        )
        self.embeddings.append(list(embedding))

    def search(self, query_embedding, top_k=5, **kwargs):
        """
        Simple nearest search using squared distance.
        """
        if not self.chunks:
            return []

        results = []

        for i, (chunk, vector) in enumerate(zip(self.chunks, self.embeddings, strict=False)):
            distance = sum(
                (a - b) ** 2
                for a, b in zip(query_embedding, vector, strict=False)
            )

            results.append(
                {
                    "chunk": chunk,
                    "score": distance,
                }
            )

        results.sort(key=lambda x: x["score"])

        return results[:top_k]
