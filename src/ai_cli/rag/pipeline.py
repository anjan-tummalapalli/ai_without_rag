from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List


@dataclass
class Chunk:
    text: str
    source: str
    start: int
    end: int
    metadata: dict


class SemanticChunker:
    """
    Sentence-aware, token-approximate chunker with overlap.
    - max_tokens: approximate maximum tokens per chunk (counts by whitespace).
    - overlap_tokens: number of tokens to overlap between adjacent chunks.
    - sentence_splitter: basic regex-based sentence splitter; replace with spaCy/NLTK for higher quality.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        # regex to split sentences (simple heuristic)
        self._sent_split_re = re.compile(r'(?<=[.!?])\s+')

    def _sentences(self, text: str) -> List[str]:
        # fallback single-split if text is short / no punctuation
        sents = [s.strip() for s in self._sent_split_re.split(text) if s.strip()]
        if not sents:
            return [text.strip()]
        return sents

    def _token_count(self, text: str) -> int:
        # approximate tokens by whitespace. Replace with actual tokenizer if needed.
        return len(text.split())

    def chunk_text(self, text: str, source: str) -> List[Chunk]:
        sents = self._sentences(text)
        chunks: List[Chunk] = []

        cur_tokens = 0
        cur_text_parts: List[str] = []
        cur_start = 0  # approximate character index

        char_cursor = 0
        for sent in sents:
            sent_tokens = self._token_count(sent)
            if cur_tokens + sent_tokens <= self.max_tokens or not cur_text_parts:
                cur_text_parts.append(sent)
                cur_tokens += sent_tokens
            else:
                chunk_text = " ".join(cur_text_parts).strip()
                chunk_end = char_cursor  # approximate; char positions not exact
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        source=source,
                        start=cur_start,
                        end=chunk_end,
                        metadata={},
                    )
                )
                # prepare next chunk with overlap
                # compute overlap by token count from the end of cur_text_parts
                overlapped = []
                overlap_count = 0
                # iterate reversed sentences and add until overlap_tokens reached
                for back_sent in reversed(cur_text_parts):
                    tcount = self._token_count(back_sent)
                    if overlap_count + tcount > self.overlap_tokens:
                        break
                    overlapped.insert(0, back_sent)
                    overlap_count += tcount
                cur_text_parts = overlapped + [sent]
                cur_tokens = sum(self._token_count(s) for s in cur_text_parts)
                cur_start = chunk_end
            char_cursor += len(sent) + 1

        # flush final chunk
        if cur_text_parts:
            chunk_text = " ".join(cur_text_parts).strip()
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    start=cur_start,
                    end=char_cursor,
                    metadata={},
                )
            )

        # add simple metadata (can be extended)
        for i, c in enumerate(chunks):
            c.metadata = {"chunk_id": i, "source": c.source, "start": c.start, "end": c.end}

        return chunks