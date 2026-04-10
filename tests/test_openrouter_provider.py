from unittest.mock import patch

from multi_llm.config import ProviderConfig
from multi_llm.providers.openrouter import OpenRouterProvider, normalize_openrouter_response
from multi_llm.types import ChatMessage


class MockResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_openrouter_provider_chat_success() -> None:
    config = ProviderConfig(
        provider="openrouter",
        api_key="test-key",
        model="openai/gpt-4.1-mini",
    )
    provider = OpenRouterProvider(config)

    with patch("multi_llm.providers.openrouter.requests.post") as post:
        post.return_value = MockResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "hello from openrouter",
                        }
                    }
                ]
            },
        )

        output = provider.chat([ChatMessage(role="user", content="hi")])

    assert output == "hello from openrouter"
    assert post.call_count == 1
    assert provider.config.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_response_normalizes_empty_tool_arguments() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"arguments": ""}},
                        {"function": {"arguments": {"name": "value"}}},
                    ]
                }
            }
        ]
    }

    normalized = normalize_openrouter_response(payload)

    tool_calls = normalized["choices"][0]["message"]["tool_calls"]
    assert tool_calls[0]["function"]["arguments"] == "{}"
    assert tool_calls[1]["function"]["arguments"] == '{"name": "value"}'
