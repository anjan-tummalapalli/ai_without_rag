# /Users/anjan/Documents/New project/ai_chat/ai_cli/src/ai_cli/rag/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional
import uuid


@dataclass(slots=True)
class Document:
    """
    Represents a loaded source document.
    """

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


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
    return (prefix + "-" if prefix else "") + uuid.uuid4().hex


def chunks_from_iterable(texts: Iterable[str], source: str) -> List[Chunk]:
    """
    Convenience: make chunks from an iterable of strings with generated ids.
    """
    return [
        Chunk(id=make_chunk_id(), text=t, source=source, chunk_index=i)
        for i, t in enumerate(texts)
    ]