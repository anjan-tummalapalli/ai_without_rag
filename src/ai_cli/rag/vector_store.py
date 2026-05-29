index.faiss
metadata.json

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from ai_cli.config.rag_config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from ai_cli.rag.models import Chunk


class VectorStore:
    """
    FAISS-backed vector database.
    """

    def __init__(self) -> None:
        self.index = None
        self.chunks: list[Chunk] = []

    def create_index(
        self,
        dimension: int,
    ) -> None:
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(
        self,
        embeddings,
        chunks: list[Chunk],
    ) -> None:
        """
        Add vectors and chunks.
        """

        vectors = np.array(embeddings).astype("float32")

        if self.index is None:
            self.create_index(vectors.shape[1])

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ):
        """
        Similarity search.
        """

        if self.index is None:
            return []

        query = np.array([query_embedding]).astype("float32")

        distances, indices = self.index.search(query, top_k)

        results = []

        for score, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "score": float(score),
                    }
                )

        return results

    def save(self) -> None:
        """
        Persist FAISS index and metadata.
        """

        if self.index is None:
            return

        faiss.write_index(
            self.index,
            str(FAISS_INDEX_PATH),
        )

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

        METADATA_PATH.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """
        Load persisted index and metadata.
        """

        if not Path(FAISS_INDEX_PATH).exists():
            return

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        metadata = json.loads(
            METADATA_PATH.read_text(
                encoding="utf-8"
            )
        )

        self.chunks = [
            Chunk(
                id=item["id"],
                text=item["text"],
                source=item["source"],
                chunk_index=item["chunk_index"],
                metadata=item.get("metadata", {}),
            )
            for item in metadata
        ]