from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class RetrievalResult:
    """
    Represents retrieved chunk information.
    """

    chunk: Chunk
    score: float