from abc import ABC, abstractmethod
from typing import Any

from multi_llm.config import ProviderConfig
from multi_llm.types import ChatMessage, ProviderCapabilities


class LLMProvider(ABC):
    """Provider-agnostic interface inspired by n8n chat model abstractions."""

    name: str

    def __init__(self, config: ProviderConfig, capabilities: ProviderCapabilities) -> None:
        self.config = config
        self.capabilities = capabilities

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        raise NotImplementedError
