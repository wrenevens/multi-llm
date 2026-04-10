# multi-llm

A lightweight Python gateway for multiple LLM providers with a unified interface.

## Why

This project follows n8n-inspired design patterns:

- provider-agnostic abstraction for chat models
- central supplier/factory for provider routing
- configuration mapping per provider
- capability flags for higher-level flows

## Install

```bash
pip install -e .
pip install -e .[dev]
```

## Quick Start

```python
from multi_llm import MultiLLM, ProviderConfig

gateway = MultiLLM.from_config(
	ProviderConfig(
		provider="openai",
		api_key="your-api-key",
		model="gpt-4o-mini",
	)
)

result = gateway.chat([
	{"role": "system", "content": "You are concise."},
	{"role": "user", "content": "Say hello."},
])

print(result)
```

## Environment Configuration

```bash
export MULTI_LLM_PROVIDER=openai
export MULTI_LLM_API_KEY=your-api-key
export MULTI_LLM_MODEL=gpt-4o-mini
# Optional:
# export MULTI_LLM_BASE_URL=https://api.openai.com/v1
# export MULTI_LLM_TIMEOUT=30
# export MULTI_LLM_MAX_RETRIES=2
```

```python
from multi_llm import MultiLLM

gateway = MultiLLM.from_env()
response = gateway.chat([
	{"role": "user", "content": "Hello"}
])
```

## Providers

- `openai` (OpenAI-compatible)
- `openrouter` (OpenAI-compatible with OpenRouter-specific normalization)
- `groq` (dedicated provider on top of OpenAI-compatible API)
- `together` (OpenAI-compatible)
- `gemini`
- `anthropic`

## Extending With New Providers

You can register a provider class at runtime using the same centralized supplier pattern.

```python
from multi_llm import ProviderConfig, register_provider, create_provider
from multi_llm.providers.base import LLMProvider
from multi_llm.types import ChatMessage, ProviderCapabilities


class MyProvider(LLMProvider):
	name = "my-provider"

	def __init__(self, config: ProviderConfig) -> None:
		super().__init__(
			config=config,
			capabilities=ProviderCapabilities(
				supports_streaming=True,
			),
		)

	def chat(self, messages: list[ChatMessage], **kwargs) -> str:
		return "hello"


register_provider("my-provider", MyProvider, default_base_url="https://api.example.com/v1")

provider = create_provider(
	ProviderConfig(provider="my-provider", api_key="k", model="my-model")
)
```

## Tests

```bash
pytest -q
```

There is at least one provider-focused test for each implemented provider.
