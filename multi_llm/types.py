from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str

    def to_openai_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_streaming: bool = False
    supports_tools: bool = False
    supports_json_mode: bool = False
