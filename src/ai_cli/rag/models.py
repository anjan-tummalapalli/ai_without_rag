# /Users/anjan/Documents/New project/ai_chat/ai_cli/src/ai_cli/rag/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Iterator
import uuid
import re


@dataclass(slots=True)
class Document:
    """
    Represents a loaded source document.
    """

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a str")
        if not isinstance(self.source, str):
            raise TypeError("source must be a str")

    def split_into_chunks(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_whole_words: bool = True,
    ) -> List["Chunk"]:
        """
        Split the document.content into Chunk objects.

        - chunk_size: approximate number of characters per chunk.
        - chunk_overlap: number of overlapping characters between consecutive chunks.
        - preserve_whole_words: if True, prefer to break at whitespace to avoid cutting words.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        chunks_texts = list(_chunk_text(self.content, chunk_size, chunk_overlap, preserve_whole_words))
        return [
            Chunk(
                id=make_chunk_id(prefix=self.source),
                text=text,
                source=self.source,
                chunk_index=i,
                metadata=dict(self.metadata),  # shallow copy so per-chunk changes don't affect original
            )
            for i, text in enumerate(chunks_texts)
        ]


@dataclass(slots=True)
class Chunk:
    """
    Represents a semantic text chunk.
    """

    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string")
        if not isinstance(self.text, str):
            raise TypeError("text must be a str")
        if not isinstance(self.source, str):
            raise TypeError("source must be a str")
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative int")


@dataclass(slots=True)
class Embedding:
    """
    Embedding vector for a piece of text.
    """

    vector: list[float]


@dataclass(slots=True)
class VectorRecord:
    """
    Record stored in a vector DB: holds embedding and the original chunk.
    """

    id: str
    embedding: Embedding
    chunk: Chunk
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    """
    Represents retrieved chunk information.
    """

    chunk: Chunk
    score: float


def make_chunk_id(prefix: Optional[str] = None) -> str:
    """
    Generate a reasonably readable chunk id. If prefix is provided, sanitize it
    (keep alphanumerics and replace others with '-') and prefix the uuid with it.
    """
    uid = uuid.uuid4().hex
    if not prefix:
        return uid
    # sanitize prefix: keep letters, numbers, replace others with hyphen, collapse multiple hyphens
    sanitized = re.sub(r"[^A-Za-z0-9]+", "-", prefix).strip("-").lower()
    sanitized = re.sub(r"-{2,}", "-", sanitized)
    # limit prefix length so id doesn't become too long
    max_prefix = 40
    sanitized = sanitized[:max_prefix]
    return f"{sanitized}-{uid}"


def _chunk_text(
    text: str, chunk_size: int, chunk_overlap: int, preserve_whole_words: bool
) -> Iterator[str]:
    """
    Yield chunks of `text` using character-based sliding window with optional overlap.
    If preserve_whole_words is True, attempts to cut at the previous whitespace boundary.
    """
    text_len = len(text)
    if text_len == 0:
        return
    start = 0
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size  # fallback, though checks above should prevent this

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            yield text[start:text_len]
            break

        if preserve_whole_words:
            # try to find last whitespace between start and end
            segment = text[start:end]
            m = re.search(r"\s+(?!.*\s+)", segment)  # find last whitespace in segment
            if m:
                # cut at the last whitespace position relative to start
                cut = start + m.start()
                # ensure we don't create an empty chunk
                if cut > start:
                    yield text[start:cut].strip()
                    start = cut - chunk_overlap
                    if start < 0:
                        start = 0
                    continue
        # default: hard cut
        yield text[start:end]
        start += step


def chunks_from_iterable(texts: Iterable[str], source: str, id_prefix: Optional[str] = None) -> List[Chunk]:
    """
    Convenience: make chunks from an iterable of strings with generated ids.
    Optional id_prefix allows source-based ids.
    """
    if not isinstance(source, str):
        raise TypeError("source must be a str")
    return [
        Chunk(id=make_chunk_id(prefix=id_prefix or source), text=t, source=source, chunk_index=i)
        for i, t in enumerate(texts)
    ]