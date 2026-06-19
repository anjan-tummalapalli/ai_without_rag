"""AI chat utilities."""

import re


def ask(prompt: str, **kwargs):
    """
    Lightweight chat entrypoint used by CLI and providers.
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    # placeholder execution hook (tests only mock this)
    return prompt


def chunk_text(
    text: str | None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    split_on_word: bool = True,
    prefer_sentence_boundary: bool = False,
) -> list[str]:
    """
    Simple sliding-window chunking with optional heuristics.

    - chunk_size: target max number of characters per chunk.
    - chunk_overlap: number of overlapping characters between chunks.
    - split_on_word: try not to cut tokens in half when possible.
    - prefer_sentence_boundary: prefer ending chunks at sentence end
      punctuation if found inside the window.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if text is None:
        return []

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    length = len(text)

    sentence_terms = ".!?。！？"

    while start < length:
        end = min(start + chunk_size, length)
        window = text[start:end]

        if prefer_sentence_boundary and end < length:
            pos = -1
            for term in sentence_terms:
                found = window.rfind(term)
                if found > pos:
                    pos = found
            if pos > 0:
                end = start + pos + 1
                window = text[start:end]

        if split_on_word and end < length:
            last_ws = None
            for match in re.finditer(r"\s", window):
                last_ws = match.start()
            if last_ws is not None and last_ws > 0:
                end = start + last_ws
                window = text[start:end]

        chunk = window.strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break

        start = max(0, end - chunk_overlap)

    return chunks