from unittest.mock import patch

from multi_llm.config import ProviderConfig
from multi_llm.providers.gemini import GeminiProvider
from multi_llm.types import ChatMessage


class MockResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_gemini_provider_chat_success() -> None:
    config = ProviderConfig(
        provider="gemini",
        api_key="test-key",
        model="gemini-2.5-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta",
    )
    provider = GeminiProvider(config)

    with patch("multi_llm.providers.gemini.requests.post") as post:
        post.return_value = MockResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "hello from gemini",
                                }
                            ]
                        }
                    }
                ]
            },
        )

        output = provider.chat(
            [
                ChatMessage(role="system", content="be concise"),
                ChatMessage(role="user", content="hi"),
            ]
        )

    assert output == "hello from gemini"
    assert post.call_count == 1
