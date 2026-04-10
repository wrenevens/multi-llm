from collections.abc import Callable
from dataclasses import dataclass

from multi_llm.config import ProviderConfig
from multi_llm.exceptions import ConfigurationError
from multi_llm.providers.anthropic import AnthropicProvider
from multi_llm.providers.base import LLMProvider
from multi_llm.providers.gemini import GeminiProvider
from multi_llm.providers.groq import GroqProvider
from multi_llm.providers.openai_compatible import OpenAICompatibleProvider


ProviderBuilder = Callable[[ProviderConfig], LLMProvider]


@dataclass(frozen=True)
class ProviderRegistration:
    builder: ProviderBuilder
    default_base_url: str | None = None


_PROVIDER_REGISTRY: dict[str, ProviderRegistration] = {}


def register_provider(
    provider: str,
    builder: ProviderBuilder,
    default_base_url: str | None = None,
    *,
    overwrite: bool = False,
) -> None:
    provider_key = provider.strip().lower()
    if not provider_key:
        raise ConfigurationError("Provider key cannot be empty")

    if provider_key in _PROVIDER_REGISTRY and not overwrite:
        raise ConfigurationError(f"Provider already registered: {provider}")

    _PROVIDER_REGISTRY[provider_key] = ProviderRegistration(
        builder=builder,
        default_base_url=default_base_url,
    )


def available_providers() -> list[str]:
    return sorted(_PROVIDER_REGISTRY.keys())


def _register_builtins() -> None:
    register_provider("openai", OpenAICompatibleProvider, "https://api.openai.com/v1", overwrite=True)
    register_provider("openrouter", OpenAICompatibleProvider, "https://openrouter.ai/api/v1", overwrite=True)
    register_provider("together", OpenAICompatibleProvider, "https://api.together.xyz/v1", overwrite=True)
    register_provider("groq", GroqProvider, "https://api.groq.com/openai/v1", overwrite=True)
    register_provider("anthropic", AnthropicProvider, "https://api.anthropic.com/v1", overwrite=True)
    register_provider("gemini", GeminiProvider, "https://generativelanguage.googleapis.com/v1beta", overwrite=True)


_register_builtins()


def create_provider(config: ProviderConfig) -> LLMProvider:
    provider_key = config.provider.strip().lower()
    registration = _PROVIDER_REGISTRY.get(provider_key)
    if not registration:
        raise ConfigurationError(
            f"Unsupported provider: {config.provider}. Available: {', '.join(available_providers())}"
        )

    if not config.base_url and registration.default_base_url:
        config.base_url = registration.default_base_url

    return registration.builder(config)
