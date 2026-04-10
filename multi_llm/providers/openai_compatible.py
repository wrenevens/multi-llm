from typing import Any

import requests

from multi_llm.config import ProviderConfig
from multi_llm.exceptions import ProviderError
from multi_llm.providers.base import LLMProvider
from multi_llm.types import ChatMessage, ProviderCapabilities


class OpenAICompatibleProvider(LLMProvider):
    """Provider for OpenAI-compatible chat completions APIs."""

    name = "openai-compatible"

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
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": kwargs.pop("model", self.config.model),
            "messages": [message.to_openai_dict() for message in messages],
            **self.config.default_params,
            **kwargs,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
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
                choices = data.get("choices") or []
                if not choices:
                    raise ProviderError("Provider returned no choices")

                message = choices[0].get("message") or {}
                content = message.get("content")
                if not isinstance(content, str):
                    raise ProviderError("Provider returned invalid content")
                return content
            except (requests.RequestException, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break

        raise ProviderError(f"OpenAI-compatible call failed after retries: {last_error}")
