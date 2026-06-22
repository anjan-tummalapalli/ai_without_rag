from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Callable


@dataclass
class Chunk:
    text: str
    start_word: int
    end_word: int  # exclusive
    index: int


class TextChunker:
    SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        tokens_per_chunk: int = 200,
        overlap: int = 50,
        tokenizer: Callable[[str], list[str]] | None = None,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        max_chunks: int | None = None,
    ) -> None:
        self.tokens_per_chunk = max(1, tokens_per_chunk)
        self.overlap = max(0, overlap)
        self.tokenizer = tokenizer or (lambda t: t.split())
        self.sentence_splitter = sentence_splitter or (
            lambda t: [s.strip() for s in self.SENTENCE_SPLIT_RE.split(t) if s.strip()]
        )
        self.max_chunks = max_chunks

    def chunk_by_tokens_with_meta(self, text: str) -> list[Chunk]:
        tokens = self.tokenizer(text)
        if not tokens:
            return []
        chunks: list[Chunk] = []
        i = 0
        step = max(1, self.tokens_per_chunk - self.overlap)
        idx = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.tokens_per_chunk]
            chunk_text = " ".join(chunk_tokens)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_word=i,
                    end_word=i + len(chunk_tokens),
                    index=idx,
                )
            )
            idx += 1
            if self.max_chunks and idx >= self.max_chunks:
                break
            i += step
        return chunks

    def chunk_by_tokens(self, text: str) -> list[str]:
        return [c.text for c in self.chunk_by_tokens_with_meta(text)]

    def chunk_by_sentences_with_meta(self, text: str, max_tokens: int | None = None) -> list[Chunk]:
        max_tokens = max_tokens or self.tokens_per_chunk
        sentences = self.sentence_splitter(text)
        if not sentences:
            return []

        sent_tokens = [self.tokenizer(s) for s in sentences]
        chunks: list[Chunk] = []
        current_tokens: list[str] = []
        current_start_idx = 0
        total_word_index = 0
        chunk_index = 0

        for tokens in sent_tokens:
            sent_len = len(tokens)
            if not current_tokens:
                current_start_idx = total_word_index

            if current_tokens and len(current_tokens) + sent_len > max_tokens:
                chunk_text = " ".join(current_tokens)
                chunks.append(Chunk(text=chunk_text, start_word=current_start_idx, end_word=current_start_idx + len(current_tokens), index=chunk_index))
                chunk_index += 1
                if self.max_chunks and chunk_index >= self.max_chunks:
                    return chunks
                overlap_count = min(self.overlap, len(current_tokens))
                overlap_tokens = current_tokens[-overlap_count:] if overlap_count > 0 else []
                current_tokens = overlap_tokens + tokens
                current_start_idx = total_word_index - overlap_count
            else:
                current_tokens.extend(tokens)

            total_word_index += sent_len

        if current_tokens:
            chunk_text = " ".join(current_tokens)
            chunks.append(Chunk(text=chunk_text, start_word=current_start_idx, end_word=current_start_idx + len(current_tokens), index=chunk_index))

        return chunks

    def chunk_by_sentences(self, text: str, max_tokens: int | None = None) -> list[str]:
        return [c.text for c in self.chunk_by_sentences_with_meta(text, max_tokens=max_tokens)]


class PromptCorrector:
    def __init__(self, tokens_per_chunk: int = 200, overlap: int = 50, **kwargs) -> None:
        self.chunker = TextChunker(tokens_per_chunk=tokens_per_chunk, overlap=overlap, **kwargs)

    def correct(self, prompt: str, by_sentences: bool = True, max_tokens: int | None = None) -> str:
        if not prompt:
            return prompt
        if by_sentences:
            chunks = self.chunker.chunk_by_sentences(prompt, max_tokens=max_tokens)
        else:
            chunks = self.chunker.chunk_by_tokens(prompt)
        return "\n\n".join(chunks)

    def correct_with_meta(self, prompt: str, by_sentences: bool = True, max_tokens: int | None = None) -> list[Chunk]:
        if not prompt:
            return []
        if by_sentences:
            return self.chunker.chunk_by_sentences_with_meta(prompt, max_tokens=max_tokens)
        return self.chunker.chunk_by_tokens_with_meta(prompt)


def prompt_corrector(prompt: str, *, tokens_per_chunk: int = 200, overlap: int = 50, by_sentences: bool = True, max_tokens: int | None = None) -> str:
    pc = PromptCorrector(tokens_per_chunk=tokens_per_chunk, overlap=overlap)
    return pc.correct(prompt, by_sentences=by_sentences, max_tokens=max_tokens)


def correct_prompt(prompt: str) -> str:
    """
    Backwards-compatible wrapper expected by tests.
    """
    if not prompt:
        return prompt
    return prompt_corrector(prompt)