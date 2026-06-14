from __future__ import annotations

import uuid
from collections.abc import Callable

from ai_cli.config.rag_config import CHUNK_OVERLAP, CHUNK_SIZE
from ai_cli.rag.models import Chunk


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        tokenizer: Callable[[str], list[str]] | None = None,
        detokenizer: Callable[[list[str]], str] | None = None,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer or (lambda tokens: " ".join(tokens))

    def chunk_text(self, text: str, source: str) -> list[Chunk]:
        tokens = self.tokenizer(text) if self.tokenizer else text.split()
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[Chunk] = []
        for idx, start in enumerate(range(0, len(tokens), step)):
            end = start + self.chunk_size
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=self.detokenizer(tokens[start:end]),
                    source=source,
                    chunk_index=idx,
                )
            )
        return chunks

# backward compatibility
def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    return SemanticChunker(chunk_size=chunk_size, overlap=overlap).chunk_text(text, source)
