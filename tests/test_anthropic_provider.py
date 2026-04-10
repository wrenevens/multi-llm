from unittest.mock import patch

from multi_llm.config import ProviderConfig
from multi_llm.providers.anthropic import AnthropicProvider
from multi_llm.types import ChatMessage


class MockResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_anthropic_provider_chat_success() -> None:
    config = ProviderConfig(
        provider="anthropic",
        api_key="test-key",
        model="claude-3-5-sonnet-latest",
        base_url="https://api.anthropic.com/v1",
    )
    provider = AnthropicProvider(config)

    with patch("multi_llm.providers.anthropic.requests.post") as post:
        post.return_value = MockResponse(
            200,
            {"content": [{"type": "text", "text": "hello from anthropic"}]},
        )

        output = provider.chat(
            [
                ChatMessage(role="system", content="be concise"),
                ChatMessage(role="user", content="hi"),
            ]
        )

    assert output == "hello from anthropic"
    assert post.call_count == 1
