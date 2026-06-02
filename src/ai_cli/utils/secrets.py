from typing import List
import re


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Simple, robust chunker:
    - Normalizes whitespace
    - Produces chunks of up to chunk_size characters with overlap
    - Ensures chunk_size > overlap
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    text = re.sub(r"\s+", " ", text).strip()
    n = len(text)
    if n == 0:
        return []
    if n <= chunk_size:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end) - overlap + overlap  # safe advance
        # simpler: move window with overlap
        start = end - overlap
    return chunks
