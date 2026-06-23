import re


def chunk_text(text: str, chunk_size: int = 1000,
               overlap: int | None = None) -> list[str]:
    """
    Robust text chunker.

    - Normalizes whitespace.
    - Produces chunks up to `chunk_size` characters with `overlap`.
    - Validates parameters and avoids infinite loops.
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive int")
    if overlap is None:
        overlap = min(200, chunk_size // 2)
    if not isinstance(overlap, int) or overlap < 0:
        raise ValueError("overlap must be a non-negative int")
    if chunk_size < overlap:
        raise ValueError("chunk_size must be greater than overlap")

    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n == 0:
        return []
    if n <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        next_start = end - overlap
        # Ensure we always make forward progress.
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks