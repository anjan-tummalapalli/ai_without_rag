# ai_cli/rag/chunking.py
from typing import List
import re


def chunk_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        split_on_word: bool = True,
        prefer_sentence_boundary: bool = False,
) -> List[str]:
        """
        Simple sliding-window chunking with a few optional heuristics.

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

        chunks: List[str] = []
        start = 0
        length = len(text)

        # Characters considered as sentence terminators. This is intentionally
        # small and can be extended for other languages.
        sentence_terms = ".!?。！？"

        while start < length:
                end = min(start + chunk_size, length)
                window = text[start:end]

                # Prefer to end at a sentence boundary if requested.
                if prefer_sentence_boundary and end < length:
                        pos = -1
                        for term in sentence_terms:
                                p = window.rfind(term)
                                if p > pos:
                                        pos = p
                        if pos > 0:
                                # include the terminator in the chunk
                                end = start + pos + 1
                                window = text[start:end]

                # Otherwise try to avoid splitting in the middle of a word.
                if split_on_word and end < length:
                        # find last whitespace in the window
                        last_ws = None
                        for m in re.finditer(r"\s", window):
                                last_ws = m.start()
                        if last_ws is not None and last_ws > 0:
                                end = start + last_ws
                                window = text[start:end]

                chunk = window.strip()
                if chunk:
                        chunks.append(chunk)

                if end >= length:
                        break
                # Move start forward but keep the desired overlap.
                start = max(0, end - chunk_overlap)

        return chunks
