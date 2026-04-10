import os
from typing import Any

from multi_llm.config import ProviderConfig
from multi_llm.exceptions import ConfigurationError
from multi_llm.factory import create_provider
from multi_llm.providers.base import LLMProvider
from multi_llm.types import ChatMessage


class MultiLLM:
    """Simple gateway facade that routes chat requests to the configured provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    @classmethod
    def from_config(cls, config: ProviderConfig) -> "MultiLLM":
        return cls(create_provider(config))

    @classmethod
    def from_env(cls, prefix: str = "MULTI_LLM_") -> "MultiLLM":
        provider = os.getenv(f"{prefix}PROVIDER")
        api_key = os.getenv(f"{prefix}API_KEY")
        model = os.getenv(f"{prefix}MODEL")
        base_url = os.getenv(f"{prefix}BASE_URL")

        if not provider or not api_key or not model:
            raise ConfigurationError(
                "Missing env vars. Expected MULTI_LLM_PROVIDER, MULTI_LLM_API_KEY, MULTI_LLM_MODEL"
            )

        timeout = float(os.getenv(f"{prefix}TIMEOUT", "30"))
        max_retries = int(os.getenv(f"{prefix}MAX_RETRIES", "2"))

        config = ProviderConfig(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        return cls.from_config(config)

    def chat(self, messages: list[dict[str, str] | ChatMessage], **kwargs: Any) -> str:
        normalized = [
            m if isinstance(m, ChatMessage) else ChatMessage(role=m["role"], content=m["content"])
            for m in messages
        ]
        return self.provider.chat(normalized, **kwargs)
