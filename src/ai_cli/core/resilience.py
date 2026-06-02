from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks with a given size and overlap.
    Keeps words together when possible.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    # Build chunks by words until approx chunk_size characters
    while start < len(words):
        current = []
        length = 0
        i = start
        while i < len(words) and (length + len(words[i]) + (1 if current else 0)) <= chunk_size:
            if current:
                length += 1  # space
            length += len(words[i])
            current.append(words[i])
            i += 1
        if not current:
            # single word longer than chunk_size: force split on characters
            word = words[start]
            for j in range(0, len(word), chunk_size - 1):
                chunks.append(word[j : j + chunk_size])
            start += 1
        else:
            chunks.append(" ".join(current))
            # advance with overlap in words
            if i >= len(words):
                break
            # find next start index so that overlap characters roughly match
            # simple approach: move back overlap words (best-effort)
            back = 0
            chars = 0
            k = i - 1
            while k >= start and chars < overlap:
                chars += len(words[k]) + (1 if chars else 0)
                back += 1
                k -= 1
            start = max(start + 1, i - back)
    return chunks
