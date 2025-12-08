"""Provider registry for managing LLM provider factories.

This module provides a registry system for managing LLM provider factories,
allowing custom providers to be registered at runtime while maintaining
backward compatibility with existing providers.

Example:
    Register a custom provider::

        from adaptive_router.core.provider_registry import default_registry
        from my_custom_provider import MyCustomLLM

        # Register custom provider
        default_registry.register("my-provider", MyCustomLLM)

        # Use in training
        trainer = Trainer(
            models=[
                Model(provider="my-provider", model_name="custom-model-1", ...)
            ],
            provider_configs={"my-provider": ProviderConfig(...)}
        )
"""

import logging
from typing import Dict, List, Type

from deepeval.models import DeepEvalBaseLLM

from ..evaluators import (
    AmazonBedrockModel,
    AnthropicModel,
    AzureOpenAIModel,
    DeepSeekModel,
    GeminiModel,
    GPTModel,
    GrokModel,
    KimiModel,
    LiteLLMModel,
    LocalModel,
    OllamaModel,
)

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Registry for managing LLM provider factories.

    This class manages a mapping of provider aliases to their corresponding
    DeepEval LLM factory classes. It allows runtime registration of custom
    providers while maintaining a set of default providers.

    Attributes:
        _factories: Dictionary mapping provider aliases to factory classes.
    """

    def __init__(self) -> None:
        """Initialize an empty provider registry."""
        self._factories: Dict[str, Type[DeepEvalBaseLLM]] = {}

    def register(
        self, alias: str, factory: Type[DeepEvalBaseLLM], force: bool = False
    ) -> None:
        """Register a provider factory.

        Args:
            alias: The provider alias (e.g., "openai", "anthropic").
            factory: The DeepEval LLM factory class.
            force: If True, overwrite existing registration. Defaults to False.

        Raises:
            ValueError: If alias is already registered and force=False.
            TypeError: If factory is not a subclass of DeepEvalBaseLLM.

        Example:
            >>> registry.register("my-provider", MyCustomLLM)
            >>> registry.register("openai", GPTModel, force=True)  # Overwrite
        """
        if not isinstance(factory, type) or not issubclass(factory, DeepEvalBaseLLM):
            raise TypeError(
                f"Factory must be a subclass of DeepEvalBaseLLM, got {type(factory)}"
            )

        if alias in self._factories and not force:
            raise ValueError(
                f"Provider '{alias}' is already registered. "
                f"Use force=True to overwrite."
            )

        self._factories[alias] = factory
        logger.info(f"Registered provider factory: {alias} -> {factory.__name__}")

    def get_factory(self, alias: str) -> Type[DeepEvalBaseLLM]:
        """Get a provider factory by alias.

        Args:
            alias: The provider alias to look up.

        Returns:
            The DeepEval LLM factory class.

        Raises:
            ValueError: If the provider alias is not registered.

        Example:
            >>> factory = registry.get_factory("openai")
            >>> client = factory(model="gpt-4", api_key="...")
        """
        factory = self._factories.get(alias)
        if factory is None:
            available = ", ".join(sorted(self._factories.keys()))
            raise ValueError(
                f"Unsupported provider: '{alias}'. Available providers: {available}"
            )
        return factory

    def is_registered(self, alias: str) -> bool:
        """Check if a provider alias is registered.

        Args:
            alias: The provider alias to check.

        Returns:
            True if the provider is registered, False otherwise.

        Example:
            >>> if registry.is_registered("openai"):
            ...     print("OpenAI is supported")
        """
        return alias in self._factories

    def list_providers(self) -> List[str]:
        """List all registered provider aliases.

        Returns:
            Sorted list of registered provider aliases.

        Example:
            >>> providers = registry.list_providers()
            >>> print(f"Supported providers: {', '.join(providers)}")
        """
        return sorted(self._factories.keys())

    def unregister(self, alias: str) -> None:
        """Remove a provider from the registry.

        Args:
            alias: The provider alias to remove.

        Raises:
            ValueError: If the provider alias is not registered.

        Example:
            >>> registry.unregister("my-old-provider")
        """
        if alias not in self._factories:
            raise ValueError(f"Provider '{alias}' is not registered")

        del self._factories[alias]
        logger.info(f"Unregistered provider factory: {alias}")

    def clear(self) -> None:
        """Remove all providers from the registry.

        Warning:
            This will remove all registered providers including defaults.
            Use with caution.

        Example:
            >>> registry.clear()
            >>> # Re-register only needed providers
        """
        self._factories.clear()
        logger.warning("Cleared all provider factories from registry")


# Create default registry with all standard providers
default_registry = ProviderRegistry()

# Register all default providers
_DEFAULT_PROVIDERS: Dict[str, Type[DeepEvalBaseLLM]] = {
    "openai": GPTModel,
    "azure": AzureOpenAIModel,
    "anthropic": AnthropicModel,
    "amazon-bedrock": AmazonBedrockModel,
    "deepseek": DeepSeekModel,
    "google": GeminiModel,
    "xai": GrokModel,
    "kimi": KimiModel,
    "litellm": LiteLLMModel,
    "local": LocalModel,
    "ollama": OllamaModel,
}

for alias, factory in _DEFAULT_PROVIDERS.items():
    default_registry.register(alias, factory)

logger.info(
    f"Initialized default provider registry with {len(_DEFAULT_PROVIDERS)} providers"
)
