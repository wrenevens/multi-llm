from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import requests

from multi_llm.config import ProviderConfig
from multi_llm.exceptions import ProviderError
from multi_llm.providers.base import LLMProvider
from multi_llm.types import ChatMessage, ProviderCapabilities


def _is_openai_response_with_choices(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("choices"), list)


def normalize_openrouter_response(payload: Any) -> Any:
    """Normalize OpenRouter responses so empty tool arguments become valid JSON."""

    if not _is_openai_response_with_choices(payload):
        return payload

    normalized = deepcopy(payload)
    for choice in normalized.get("choices", []):
        tool_calls = (choice.get("message") or {}).get("tool_calls") or []
        for tool_call in tool_calls:
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            args = function.get("arguments")
            if isinstance(args, str) and args.strip():
                continue

            if isinstance(args, dict):
                function["arguments"] = json.dumps(args)
            else:
                function["arguments"] = "{}"

    return normalized


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider with OpenAI-compatible transport and payload normalization."""

    name = "openrouter"

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            config.base_url = "https://openrouter.ai/api/v1"

        super().__init__(
            config=config,
            capabilities=ProviderCapabilities(
                supports_streaming=True,
                supports_tools=True,
                supports_json_mode=True,
            ),
        )

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> str:
        request_options = dict(kwargs)
        http_referer = request_options.pop("http_referer", "https://github.com/wrenevens/multi-llm")
        x_title = request_options.pop("x_title", "multi-llm")
        model = request_options.pop("model", self.config.model)

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": [message.to_openai_dict() for message in messages],
            **self.config.default_params,
            **request_options,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": http_referer,
            "X-Title": x_title,
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

                data = normalize_openrouter_response(response.json())
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

        raise ProviderError(f"OpenRouter call failed after retries: {last_error}")