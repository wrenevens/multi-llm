from unittest.mock import patch

from multi_llm.config import ProviderConfig
from multi_llm.providers.groq import GroqProvider
from multi_llm.types import ChatMessage


class MockResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


def test_groq_provider_chat_success() -> None:
    config = ProviderConfig(
        provider="groq",
        api_key="test-key",
        model="llama-3.3-70b-versatile",
    )
    provider = GroqProvider(config)

    with patch("multi_llm.providers.openai_compatible.requests.post") as post:
        post.return_value = MockResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "hello from groq",
                        }
                    }
                ]
            },
        )

        output = provider.chat([ChatMessage(role="user", content="hi")])

    assert output == "hello from groq"
    assert post.call_count == 1
    assert provider.config.base_url == "https://api.groq.com/openai/v1"
