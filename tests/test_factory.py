from multi_llm.config import ProviderConfig
from multi_llm.exceptions import ConfigurationError
from multi_llm.factory import create_provider, register_provider
from multi_llm.providers.anthropic import AnthropicProvider
from multi_llm.providers.base import LLMProvider
from multi_llm.providers.gemini import GeminiProvider
from multi_llm.providers.groq import GroqProvider
from multi_llm.providers.openai_compatible import OpenAICompatibleProvider
from multi_llm.types import ChatMessage, ProviderCapabilities


def test_factory_returns_openai_compatible_provider() -> None:
    provider = create_provider(
        ProviderConfig(provider="openrouter", api_key="k", model="x")
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.config.base_url == "https://openrouter.ai/api/v1"


def test_factory_returns_anthropic_provider() -> None:
    provider = create_provider(
        ProviderConfig(provider="anthropic", api_key="k", model="x")
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.config.base_url == "https://api.anthropic.com/v1"


def test_factory_returns_groq_provider() -> None:
    provider = create_provider(
        ProviderConfig(provider="groq", api_key="k", model="x")
    )
    assert isinstance(provider, GroqProvider)
    assert provider.config.base_url == "https://api.groq.com/openai/v1"


def test_factory_returns_gemini_provider() -> None:
    provider = create_provider(
        ProviderConfig(provider="gemini", api_key="k", model="gemini-2.5-flash")
    )
    assert isinstance(provider, GeminiProvider)
    assert provider.config.base_url == "https://generativelanguage.googleapis.com/v1beta"


def test_factory_raises_for_unknown_provider() -> None:
    try:
        create_provider(ProviderConfig(provider="unknown", api_key="k", model="x"))
    except ConfigurationError:
        assert True
        return

    assert False, "Expected ConfigurationError"


class DummyProvider(LLMProvider):
    name = "dummy"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config=config, capabilities=ProviderCapabilities())

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:  # noqa: ANN003
        return "dummy"


def test_factory_supports_registering_custom_provider() -> None:
    register_provider("dummy", DummyProvider, "https://dummy.local", overwrite=True)
    provider = create_provider(
        ProviderConfig(provider="dummy", api_key="k", model="x")
    )

    assert isinstance(provider, DummyProvider)
    assert provider.config.base_url == "https://dummy.local"
