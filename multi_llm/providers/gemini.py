from typing import Any

import requests

from multi_llm.config import ProviderConfig
from multi_llm.exceptions import ProviderError
from multi_llm.providers.base import LLMProvider
from multi_llm.types import ChatMessage, ProviderCapabilities


class GeminiProvider(LLMProvider):
    """Google Gemini provider via the Generative Language API."""

    name = "gemini"

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
        model = kwargs.pop("model", self.config.model)
        url = f"{self.config.base_url.rstrip('/')}/models/{model}:generateContent"

        system_parts = [m.content for m in messages if m.role == "system"]
        contents = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})

        generation_config = {
            **self.config.default_params,
            **kwargs,
        }

        payload: dict[str, Any] = {
            "contents": contents,
        }
        if generation_config:
            payload["generationConfig"] = generation_config
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n".join(system_parts)}],
            }

        headers = {
            "x-goog-api-key": self.config.api_key,
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
                candidates = data.get("candidates") or []
                if not candidates:
                    raise ProviderError("Provider returned no candidates")

                candidate_content = candidates[0].get("content") or {}
                parts = candidate_content.get("parts") or []
                for part in parts:
                    text = part.get("text")
                    if isinstance(text, str):
                        return text

                raise ProviderError("Provider returned no text content")
            except (requests.RequestException, ValueError, ProviderError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break

        raise ProviderError(f"Gemini call failed after retries: {last_error}")
