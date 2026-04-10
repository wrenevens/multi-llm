from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderConfig:
    provider: str
    api_key: str
    model: str
    base_url: str | None = None
    timeout: float = 30.0
    max_retries: int = 2
    default_params: dict[str, Any] = field(default_factory=dict)
