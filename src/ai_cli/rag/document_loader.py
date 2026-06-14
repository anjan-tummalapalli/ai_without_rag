from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    source: str
    metadata: dict[str, str] = field(default_factory=dict)
    chunk_id: int | None = None
