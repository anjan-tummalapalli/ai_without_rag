from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional

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

from ai_cli.config.rag_config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from ai_cli.rag.models import Chunk


class VectorStore:
    """
    FAISS-backed vector store with:
    - add / upsert / delete
    - persisted index + metadata + embeddings (.npy beside FAISS file)
    - filtered similarity search
    """
    def __init__(self) -> None:
        self.index = None
        self.chunks = []
        self.embeddings = None
        self._embeddings_path = Path(str(FAISS_INDEX_PATH)).with_suffix(".npy")

    def create_index(self, dimension: int) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss library is not available; please install faiss to use VectorStore (e.g. pip install faiss-cpu or faiss)."
            )
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks: List[Chunk] = []
        # embeddings stored in same order as self.chunks, dtype=float32
        self.embeddings: Optional[np.ndarray] = None

        # derive embeddings path from index path (same directory, .npy)
        self._embeddings_path = Path(str(FAISS_INDEX_PATH)).with_suffix(".npy")

    def _rebuild_index(self) -> None:
        """Recreate FAISS index from current embeddings (used for upsert/delete)."""
        if self.embeddings is None or len(self.embeddings) == 0:
            self.index = None
            return
        dim = int(self.embeddings.shape[1])
        self.create_index(dim)
        self.index.add(self.embeddings)

    def add_embeddings(self, embeddings: Iterable[np.ndarray], chunks: List[Chunk]) -> None:
        """
        Append new vectors and chunks.
        - embeddings: iterable of vectors shaped (dim,)
        - chunks: list of Chunk objects with unique ids for each vector (not enforced)
        """
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

    def upsert(self, embeddings: Iterable[np.ndarray], chunks: List[Chunk]) -> None:
        """
        Upsert behavior: if a chunk.id exists, replace its vector and chunk metadata;
        otherwise append as new.
        This will rebuild the index if any replacement occurs.
        """
        vectors = np.array(list(embeddings)).astype("float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # build id -> position map
        id_to_pos = {c.id: i for i, c in enumerate(self.chunks)}

        replaced = False
        new_chunks = []
        new_vectors = []

        for vec, chunk in zip(vectors, chunks):
            if chunk.id in id_to_pos:
                pos = id_to_pos[chunk.id]
                # replace in-place
                if self.embeddings is not None:
                    self.embeddings[pos] = vec
                self.chunks[pos] = chunk
                replaced = True
            else:
                new_chunks.append(chunk)
                new_vectors.append(vec)

        if new_vectors:
            new_vectors_np = np.array(new_vectors).astype("float32")
            if self.embeddings is None:
                self.embeddings = new_vectors_np.copy()
            else:
                self.embeddings = np.vstack([self.embeddings, new_vectors_np])
            self.chunks.extend(new_chunks)

        # If any replacements happened, rebuild the index to ensure integrity.
        if replaced or self.index is None:
            self._rebuild_index()
        else:
            # only append new vectors to existing index
            if new_vectors:
                self.index.add(new_vectors_np)

    def delete(self, ids: Iterable[str]) -> None:
        """
        Delete chunks and embeddings by chunk.id. Rebuilds index.
        """
        ids_to_remove = set(ids)
        if not ids_to_remove:
            return

        keep_pairs = [
            (emb, ch)
            for emb, ch in zip(self.embeddings or np.empty((0,)), self.chunks)
            if ch.id not in ids_to_remove
        ]

        if keep_pairs:
            kept_embeddings = np.vstack([p[0] for p in keep_pairs]).astype("float32")
            kept_chunks = [p[1] for p in keep_pairs]
        else:
            kept_embeddings = np.empty((0, self.embeddings.shape[1] if self.embeddings is not None else 0), dtype="float32")
            kept_chunks = []

        self.embeddings = kept_embeddings if kept_embeddings.size else None
        self.chunks = kept_chunks
        self._rebuild_index()

    def search(
        self,
        query_embedding,
        top_k: int = 5,
        filter_fn: Optional[Callable[[Chunk], bool]] = None,
        return_scores: bool = True,
    ):
        """
        Similarity search with optional filter function.
        - filter_fn: callable receiving a Chunk, returning True to keep.
        Returns list of dicts: {"chunk": Chunk, "score": float}
        """
        if self.index is None or self.embeddings is None:
            return []

        query = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if filter_fn is not None and not filter_fn(chunk):
                continue
            item = {"chunk": chunk}
            if return_scores:
                # Faiss returns L2 distance; convert to similarity score (optional).
                # Keep raw distance for now; caller can convert if needed.
                item["score"] = float(score)
            results.append(item)

        return results

    def save(self) -> None:
        """
        Persist FAISS index, metadata, and embeddings.
        """
        if self.index is None:
            # ensure directory exists and still save metadata/embeddings
            pass
        else:
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))

        # metadata
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
        Path(METADATA_PATH).parent.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # embeddings (.npy) - keep aligned with chunks
        if self.embeddings is not None:
            self._embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(self._embeddings_path), self.embeddings)

    def load(self) -> None:
        """
        Load persisted index, metadata, and embeddings if present.
        """
        faiss_path = Path(FAISS_INDEX_PATH)
        if faiss_path.exists():
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
            self.embeddings = np.load(str(self._embeddings_path)).astype("float32")
        else:
            # try to infer dimension from index if possible
            if self.index is not None:
                d = int(self.index.d)
                self.embeddings = np.empty((0, d), dtype="float32")
            else:
                self.embeddings = None

# ------------------------------------------------------------------
# Backward compatibility alias
# ------------------------------------------------------------------
InMemoryVectorStore = VectorStore