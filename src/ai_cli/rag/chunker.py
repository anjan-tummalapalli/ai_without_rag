from __future__ import annotations

import uuid
from typing import Iterable

from ai_cli.config.rag_config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
)
from ai_cli.rag.models import Chunk

class SemanticChunker:
    """
    Token-aware semantic chunker.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(
        self,
        text: str,
        source: str,
    ) -> list[Chunk]:
        """
        Split text into overlapping semantic chunks.
        """

        words = text.split()

        chunks: list[Chunk] = []

        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + self.chunk_size

            chunk_words = words[start:end]

            chunk_text = " ".join(chunk_words)

            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                )
            )

            start += self.chunk_size - self.overlap
            chunk_index += 1

        return chunks