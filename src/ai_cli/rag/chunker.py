from __future__ import annotations

import uuid
from typing import Callable, Iterable, List, Optional

from ai_cli.config.rag_config import CHUNK_OVERLAP, CHUNK_SIZE
from ai_cli.rag.models import Chunk


class SemanticChunker:
    """
    Token-aware semantic chunker.

    - By default it chunks on whitespace (word-level).
    - You can pass a tokenizer/encoder function that maps text -> list[str] tokens.
      The overlap and chunk_size are interpreted in token units if a tokenizer is provided.
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
        # detokenizer converts tokens back to text; if not provided we join by space
        self.detokenizer = detokenizer or (lambda tokens: " ".join(tokens))

    def chunk_text(self, text: str, source: str) -> List[Chunk]:
        """
        Split text into overlapping semantic chunks.

        Returns a list of Chunk objects (same shape as existing Chunk model).
        chunk_size and overlap are applied to token count if tokenizer is provided,
        otherwise applied to words (whitespace-split).
        """
        if self.tokenizer:
            tokens = self.tokenizer(text)
        else:
            # fallback to whitespace tokenizer for words
            tokens = text.split()

        chunks: List[Chunk] = []

        start = 0
        chunk_index = 0
        total = len(tokens)

        while start < total:
            end = min(start + self.chunk_size, total)
            chunk_tokens = tokens[start:end]
            chunk_text = self.detokenizer(chunk_tokens)

            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                )
            )

            # advance by chunk_size - overlap (ensure progress)
            step = max(1, self.chunk_size - self.overlap)
            start += step
            chunk_index += 1

        return chunks