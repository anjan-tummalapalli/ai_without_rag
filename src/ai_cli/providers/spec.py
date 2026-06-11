from dataclasses import dataclass
from typing import Optional, Any

@dataclass(frozen=True)
class ProviderRequest:
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    kwargs: dict[str, Any] = None
