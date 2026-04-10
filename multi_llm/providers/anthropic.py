from typing import Any

import requests

from multi_llm.config import ProviderConfig
from multi_llm.exceptions import ProviderError
from multi_llm.providers.base import LLMProvider
from multi_llm.types import ChatMessage, ProviderCapabilities


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(
            config=config,
            capabilities=ProviderCapabilities(
                supports_streaming=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
        )

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        system_messages = [m.content for m in messages if m.role == "system"]
        chat_messages = [
            {"role": m.role if m.role in {"user", "assistant"} else "user", "content": m.content}
            for m in messages
            if m.role != "system"
        ]

        payload = {
            "model": kwargs.pop("model", self.config.model),
            "messages": chat_messages,
            **self.config.default_params,
            **kwargs,
        }
        if system_messages:
            payload["system"] = "\n".join(system_messages)

        url = f"{self.config.base_url.rstrip('/')}/messages"
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
                if response.status_code >= 500:
                    raise ProviderError(f"Provider server error: {response.status_code} {response.text}")
                if response.status_code >= 400:
                    raise ProviderError(f"Provider request failed: {response.status_code} {response.text}")

                data = response.json()
                blocks = data.get("content") or []
                for block in blocks:
                    if block.get("type") == "text" and isinstance(block.get("text"), str):
                        return block["text"]

                raise ProviderError("Provider returned no text content")
            except (requests.RequestException, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break

        raise ProviderError(f"Anthropic call failed after retries: {last_error}")
