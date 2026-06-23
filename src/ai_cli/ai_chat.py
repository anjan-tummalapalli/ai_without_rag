"""Text chunking utilities."""
 
from __future__ import annotations
 
import re
 
# Compiled once at module level — avoids re-compiling on every call.
_SENTENCE_END_RE: re.Pattern[str] = re.compile(r"[.!?。！？]")
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s")
 
 
def _last_sentence_end(window: str) -> int:
    """
    Return the index of the last sentence-terminating character in *window*,
    or -1 if none is found.
 
    Uses a single compiled regex scan instead of calling ``str.rfind`` for
    each punctuation character separately.
    """
    pos = -1
    for match in _SENTENCE_END_RE.finditer(window):
        pos = match.start()
    return pos
 
 
def _last_whitespace(window: str) -> int:
    """
    Return the index of the last whitespace character in *window*, or -1 if
    none is found.
 
    Uses ``str.rfind`` over the common ASCII space first (O(n), branch-
    predicted well for typical prose), then falls back to a compiled regex scan
    for Unicode whitespace such as tabs or non-breaking spaces.  This avoids
    iterating *all* matches just to find the last one.
    """
    # Fast path: plain space covers the vast majority of real text.
    idx = window.rfind(" ")
    if idx >= 0:
        return idx
    # Slow path: tabs, non-breaking spaces, etc.
    pos = -1
    for match in _WHITESPACE_RE.finditer(window):
        pos = match.start()
    return pos
 
 
def _next_start(end: int, chunk_overlap: int, prev_start: int) -> int:
    """
    Compute the start position of the next chunk.
 
    Clamps the result so that we always advance by at least one character,
    preventing an infinite loop when ``chunk_overlap`` is large enough that
    ``end - chunk_overlap <= prev_start`` (which can happen after a heuristic
    trims ``end`` backward).
    """
    candidate = end - chunk_overlap
    return candidate if candidate > prev_start else prev_start + 1
 
 
def chunk_text(
    text: str | None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    split_on_word: bool = True,
    prefer_sentence_boundary: bool = False,
) -> list[str]:
    """
    Sliding-window text chunker with optional split heuristics.
 
    Args:
        text: Input text to chunk.  ``None`` or whitespace-only returns ``[]``.
        chunk_size: Maximum number of characters per chunk (must be > 0).
        chunk_overlap: Characters of overlap between consecutive chunks
            (must be < ``chunk_size``).
        split_on_word: When ``True``, trim the chunk boundary to the last
            whitespace position so words are not split mid-token.
        prefer_sentence_boundary: When ``True``, prefer ending a chunk at the
            last sentence-terminating character (``.!?`` etc.) found in the
            window.  Applied before ``split_on_word``.
 
    Returns:
        List of non-empty, stripped chunk strings.
 
    Raises:
        ValueError: If ``chunk_size <= 0`` or ``chunk_overlap >= chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
 
    if not text:
        return []
 
    text = text.strip()
    if not text:
        return []
 
    chunks: list[str] = []
    start = 0
    length = len(text)
 
    while start < length:
        end = min(start + chunk_size, length)
 
        if end < length:
            if prefer_sentence_boundary:
                pos = _last_sentence_end(text[start:end])
                if pos > 0:
                    end = start + pos + 1
 
            if split_on_word:
                pos = _last_whitespace(text[start:end])
                if pos > 0:
                    end = start + pos
 
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
 
        if end >= length:
            break
 
        start = _next_start(end, chunk_overlap, start)
 
    return chunks