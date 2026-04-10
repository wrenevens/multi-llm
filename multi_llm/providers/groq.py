from multi_llm.config import ProviderConfig
from multi_llm.providers.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """Dedicated Groq provider using Groq's OpenAI-compatible API."""

    name = "groq"

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            config.base_url = "https://api.groq.com/openai/v1"
        super().__init__(config)
