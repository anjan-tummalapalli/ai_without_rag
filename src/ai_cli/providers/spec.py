from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProviderRequest:
    provider: str
    model: str | None = None
    api_key: str | None = None
    kwargs: dict[str, Any] = None