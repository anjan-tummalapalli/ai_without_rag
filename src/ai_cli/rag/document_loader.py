from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class Document:
    content: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)
    chunk_id: Optional[int] = None
