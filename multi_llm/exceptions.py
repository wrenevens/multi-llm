class MultiLLMError(Exception):
    """Base exception for multi-llm errors."""


class ProviderError(MultiLLMError):
    """Raised when a provider returns an invalid response or hard failure."""


class ConfigurationError(MultiLLMError):
    """Raised when provider configuration is invalid or incomplete."""
