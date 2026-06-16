"""Token-aware sentence chunker with overlap and character spans.

For a simpler sliding-window chunker, see ``ai_cli.rag.chunker``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    start: int
    end: int
    metadata: dict


class SemanticChunker:
    """
    Improved sentence-aware, token-accurate chunker with overlap and real character spans.

    Key improvements:
    - Computes exact character start/end indices for chunks (based on the original text).
    - Splits very long sentences by token count so no single "sentence" exceeds max_tokens.
    - Performs chunking at token granularity and preserves overlap by token count.
    - Validates and clamps overlap to avoid infinite loops.
    - Allows passing a custom sentence splitter callable (optional).
    """

    SENT_RE = re.compile(r'.+?(?:[.!?](?:["\'])?\s+|$)', re.DOTALL)
    TOKEN_RE = re.compile(r'\S+')

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        sentence_splitter: Callable[[str], list[tuple[str, int, int]]] | None = None,
    ):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.max_tokens = max_tokens
        # clamp overlap so that progress is guaranteed when splitting
        self.overlap_tokens = max(0, min(overlap_tokens, max_tokens - 1))
        self._custom_splitter = sentence_splitter

    def _default_sentence_spans(self, text: str) -> list[tuple[str, int, int]]:
        """
        Returns list of (sentence_text, start_index, end_index) using a simple regex.
        Falls back to whole text if nothing matched.
        """
        spans = []
        for m in self.SENT_RE.finditer(text):
            s = m.group(0).strip()
            if not s:
                continue
            # compute trimmed span to remove leading/trailing whitespace while preserving indices
            abs_start = m.start()
            abs_end = m.end()
            # trim leading whitespace
            leading_ws = len(s) - len(s.lstrip())
            trailing_ws = len(s) - len(s.rstrip())
            sent_start = abs_start + (m.group(0).find(s))
            sent_end = sent_start + len(s)
            spans.append((s, sent_start, sent_end))
        if not spans and text.strip():
            ts = text.strip()
            start = text.find(ts)
            spans.append((ts, start, start + len(ts)))
        return spans

    def _sentence_spans(self, text: str) -> list[tuple[str, int, int]]:
        if self._custom_splitter:
            return self._custom_splitter(text)
        return self._default_sentence_spans(text)

    def _token_spans(self, text: str) -> list[tuple[int, int]]:
        """Return list of (start, end) spans for tokens across the whole text."""
        return [(m.start(), m.end()) for m in self.TOKEN_RE.finditer(text)]

    def chunk_text(self, text: str, source: str) -> list[Chunk]:
        sentences = self._sentence_spans(text)
        if not sentences:
            return []

        token_spans = self._token_spans(text)
        # precompute token index -> (start,end)
        # for fast mapping, build a list of token indices belonging to each sentence
        sent_token_indices: list[list[int]] = []
        tok_idx = 0
        N_tokens = len(token_spans)

        for _, s_start, s_end in sentences:
            indices = []
            # advance tok_idx until token_start >= s_start
            while tok_idx < N_tokens and token_spans[tok_idx][0] < s_start:
                tok_idx += 1
            j = tok_idx
            while j < N_tokens and token_spans[j][1] <= s_end:
                indices.append(j)
                j += 1
            sent_token_indices.append(indices)
            # next sentence may reuse current j as starting point
            tok_idx = j

        chunks: list[Chunk] = []
        cur_tokens_idxs: list[int] = []

        def flush_current_chunk():
            if not cur_tokens_idxs:
                return
            start = token_spans[cur_tokens_idxs[0]][0]
            end = token_spans[cur_tokens_idxs[-1]][1]
            chunk_text = text[start:end]
            chunk = Chunk(
                text=chunk_text,
                source=source,
                start=start,
                end=end,
                metadata={},
            )
            chunks.append(chunk)

        # iterate over sentence token lists, splitting oversized sentences into token groups
        for sent_indices in sent_token_indices:
            if not sent_indices:
                continue

            # break sentence token indices into groups of at most max_tokens
            i = 0
            L = len(sent_indices)
            while i < L:
                group = sent_indices[i : i + self.max_tokens]
                i += len(group)
                group_len = len(group)

                if not cur_tokens_idxs:
                    # start new chunk
                    cur_tokens_idxs.extend(group)
                    continue

                if len(cur_tokens_idxs) + group_len <= self.max_tokens:
                    cur_tokens_idxs.extend(group)
                    continue

                # need to flush current chunk and start a new one with overlap
                flush_current_chunk()
                # prepare overlap from the tail of the just-flushed token list
                # ensure we don't exceed max_tokens and overlap is clamped earlier
                overlap_count = min(self.overlap_tokens, len(cur_tokens_idxs))
                overlapped = cur_tokens_idxs[-overlap_count:] if overlap_count > 0 else []
                # start new current tokens with overlapped tokens + current group
                cur_tokens_idxs = overlapped + group

        # final flush
        flush_current_chunk()

        # add metadata
        for i, c in enumerate(chunks):
            c.metadata = {
                "chunk_id": i,
                "source": c.source,
                "start": c.start,
                "end": c.end,
                "token_count": len(self.TOKEN_RE.findall(c.text)),
            }

        return chunks

class RAGPipeline:
    """
    Base RAG pipeline interface (or minimal stub for now)
    """
    
    def __init__(self):
        pass
