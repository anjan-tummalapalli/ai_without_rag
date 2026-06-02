from __future__ import annotations
import re
from typing import Iterable, List, Optional


class TextChunker:
    """
    Simple, configurable chunker.

    - chunk_by_sentences: groups sentences into chunks with a target max_tokens (approx by words).
    - chunk_by_tokens: naive split-by-words chunker with overlap.
    """

    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
    DEFAULT_TOKEN_APPROX = 1  # 1 token ~= 1 word approximation

    def __init__(self, tokens_per_chunk: int = 200, overlap: int = 50) -> None:
        self.tokens_per_chunk = max(1, tokens_per_chunk)
        self.overlap = max(0, overlap)

    def _words(self, text: str) -> List[str]:
        return text.split()

    def chunk_by_tokens(self, text: str) -> List[str]:
        words = self._words(text)
        if not words:
            return []
        chunks = []
        i = 0
        step = max(1, self.tokens_per_chunk - self.overlap)
        while i < len(words):
            chunk_words = words[i : i + self.tokens_per_chunk]
            chunks.append(" ".join(chunk_words))
            i += step
        return chunks

    def chunk_by_sentences(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        max_tokens = max_tokens or self.tokens_per_chunk
        sentences = [s.strip() for s in self.SENTENCE_SPLIT_RE.split(text) if s.strip()]
        if not sentences:
            return []
        chunks = []
        current = []
        current_tokens = 0
        for sent in sentences:
            token_count = len(self._words(sent))
            if current and current_tokens + token_count > max_tokens:
                chunks.append(" ".join(current))
                # slide by overlap sentences approximated by overlap words - keep it simple and reset
                current = [sent]
                current_tokens = token_count
            else:
                current.append(sent)
                current_tokens += token_count
        if current:
            chunks.append(" ".join(current))
        # optionally apply overlap by merging adjacent chunks (omitted for simplicity)
        return chunks
