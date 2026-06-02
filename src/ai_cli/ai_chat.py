# ai_cli/rag/chunking.py
from typing import List

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """
        Simple whitespace-based sliding-window chunking.
        - chunk_size: approximate max number of characters per chunk
        - chunk_overlap: number of overlapping characters between consecutive chunks
        """
        if chunk_size <= 0:
                raise ValueError("chunk_size must be > 0")
        if chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")

        text = text.strip()
        if not text:
                return []

        # Use characters for chunking to keep implementation simple and deterministic.
        chunks = []
        start = 0
        length = len(text)
        while start < length:
                end = min(start + chunk_size, length)
                chunk = text[start:end].strip()
                if chunk:
                        chunks.append(chunk)
                if end == length:
                        break
                start = max(0, end - chunk_overlap)
        return chunks
