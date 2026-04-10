from unittest.mock import patch

from multi_llm.config import ProviderConfig
from multi_llm.gateway import MultiLLM


def test_gateway_chat_uses_provider() -> None:
    gateway = MultiLLM.from_config(
        ProviderConfig(
            provider="openai",
            api_key="k",
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
        )
    )

    with patch("multi_llm.providers.openai_compatible.requests.post") as post:
        post.return_value.status_code = 200
        post.return_value.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }

        result = gateway.chat([{"role": "user", "content": "hello"}])

    assert result == "ok"


def test_gateway_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MULTI_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MULTI_LLM_API_KEY", "k")
    monkeypatch.setenv("MULTI_LLM_MODEL", "gpt-4o-mini")

    gateway = MultiLLM.from_env()

    assert gateway.provider.config.provider == "openai"
