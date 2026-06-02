from __future__ import annotations

import uuid
from typing import Callable, List, Optional
from ai_cli.config.rag_config import CHUNK_OVERLAP, CHUNK_SIZE
from ai_cli.rag.models import Chunk

class SemanticChunker:
    """
    Token-aware semantic chunker.
    """
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        detokenizer: Optional[Callable[[List[str]], str]] = None,
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.overlap = int(overlap)
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer or (lambda tokens: " ".join(tokens))

    def chunk_text(self, text: str, source: str) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        """

        if self.tokenizer:
            tokens = self.tokenizer(text)
        else:
            tokens = text.split()

        chunks: List[Chunk] = []
        start = 0
        chunk_index = 0
        total = len(tokens)
        while start < total:
            end = min(start + self.chunk_size, total)
            chunk_tokens = tokens[start:end]
            chunk_body = self.detokenizer(chunk_tokens)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_body,
                    source=source,
                    chunk_index=chunk_index,
                )
            )
            step = max(1, self.chunk_size - self.overlap)
            start += step
            chunk_index += 1
        return chunks

# ------------------------------------------------------------------
# Backward compatibility export
# ------------------------------------------------------------------
def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
):
    """
    Backward-compatible helper.
    Existing code imports:
        from ai_cli.rag.chunker import chunk_text
    """
    chunker = SemanticChunker(
        chunk_size=chunk_size,
        overlap=overlap,
    )
    return chunker.chunk_text(text, source)