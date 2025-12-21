"""Tests for ProviderRegistry."""

import pytest
from deepeval.models import DeepEvalBaseLLM

from nordlys.core.provider_registry import ProviderRegistry, default_registry
from nordlys.evaluators import (
    AnthropicModel,
    GPTModel,
    GeminiModel,
    OllamaModel,
)


class MockLLMProvider(DeepEvalBaseLLM):
    """Mock LLM provider for testing."""

    def __init__(self, model: str, api_key: str = "test-key", **kwargs):
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def load_model(self):
        """Mock load_model method."""
        return self

    def generate(self, prompt: str) -> str:
        """Mock generate method."""
        return f"Mock response to: {prompt}"

    async def a_generate(self, prompt: str) -> str:
        """Mock async generate method."""
        return f"Mock async response to: {prompt}"

    def get_model_name(self) -> str:
        """Mock get_model_name method."""
        return self.model


class InvalidProvider:
    """Invalid provider that doesn't inherit from DeepEvalBaseLLM."""

    pass


@pytest.fixture
def empty_registry() -> ProviderRegistry:
    """Create an empty ProviderRegistry for testing."""
    return ProviderRegistry()


@pytest.fixture
def populated_registry() -> ProviderRegistry:
    """Create a ProviderRegistry with some providers for testing."""
    registry = ProviderRegistry()
    registry.register("openai", GPTModel)
    registry.register("anthropic", AnthropicModel)
    registry.register("google", GeminiModel)
    return registry


class TestProviderRegistryInitialization:
    """Test ProviderRegistry initialization."""

    def test_empty_initialization(self, empty_registry: ProviderRegistry) -> None:
        """Test ProviderRegistry initializes empty."""
        assert len(empty_registry.list_providers()) == 0
        assert not empty_registry.is_registered("openai")

    def test_default_registry_has_providers(self) -> None:
        """Test default_registry is pre-populated with standard providers."""
        providers = default_registry.list_providers()

        # Check all expected providers are registered
        expected_providers = [
            "amazon-bedrock",
            "anthropic",
            "azure",
            "deepseek",
            "google",
            "kimi",
            "litellm",
            "local",
            "ollama",
            "openai",
            "xai",
        ]

        assert len(providers) == len(expected_providers)
        for provider in expected_providers:
            assert provider in providers


class TestProviderRegistryRegister:
    """Test ProviderRegistry.register method."""

    def test_register_new_provider(self, empty_registry: ProviderRegistry) -> None:
        """Test registering a new provider."""
        empty_registry.register("mock-provider", MockLLMProvider)

        assert empty_registry.is_registered("mock-provider")
        assert "mock-provider" in empty_registry.list_providers()
        assert empty_registry.get_factory("mock-provider") == MockLLMProvider

    def test_register_multiple_providers(
        self, empty_registry: ProviderRegistry
    ) -> None:
        """Test registering multiple providers."""
        empty_registry.register("openai", GPTModel)
        empty_registry.register("anthropic", AnthropicModel)
        empty_registry.register("google", GeminiModel)

        providers = empty_registry.list_providers()
        assert len(providers) == 3
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers

    def test_register_duplicate_raises_error(
        self, populated_registry: ProviderRegistry
    ) -> None:
        """Test registering duplicate provider raises ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            populated_registry.register("openai", MockLLMProvider)

    def test_register_duplicate_with_force(
        self, populated_registry: ProviderRegistry
    ) -> None:
        """Test registering duplicate provider with force=True overwrites."""
        original_factory = populated_registry.get_factory("openai")
        assert original_factory == GPTModel

        # Overwrite with force=True
        populated_registry.register("openai", MockLLMProvider, force=True)

        new_factory = populated_registry.get_factory("openai")
        assert new_factory == MockLLMProvider
        assert new_factory != original_factory

    def test_register_invalid_factory_raises_error(
        self, empty_registry: ProviderRegistry
    ) -> None:
        """Test registering invalid factory raises TypeError."""
        with pytest.raises(TypeError, match="must be a subclass of DeepEvalBaseLLM"):
            empty_registry.register("invalid", InvalidProvider)  # type: ignore

    def test_register_non_class_raises_error(
        self, empty_registry: ProviderRegistry
    ) -> None:
        """Test registering non-class raises TypeError."""
        with pytest.raises(TypeError, match="must be a subclass of DeepEvalBaseLLM"):
            empty_registry.register("invalid", "not-a-class")  # type: ignore


class TestProviderRegistryGetFactory:
    """Test ProviderRegistry.get_factory method."""

    def test_get_factory_success(self, populated_registry: ProviderRegistry) -> None:
        """Test getting a registered factory."""
        factory = populated_registry.get_factory("openai")
        assert factory == GPTModel

    def test_get_factory_not_found(self, empty_registry: ProviderRegistry) -> None:
        """Test getting unregistered factory raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            empty_registry.get_factory("nonexistent")

    def test_get_factory_error_message_includes_available(
        self, populated_registry: ProviderRegistry
    ) -> None:
        """Test error message includes available providers."""
        with pytest.raises(
            ValueError, match="Available providers: anthropic, google, openai"
        ):
            populated_registry.get_factory("nonexistent")

    def test_get_factory_can_instantiate(
        self, populated_registry: ProviderRegistry
    ) -> None:
        """Test that retrieved factory can instantiate a client."""
        factory = populated_registry.get_factory("openai")

        # This would normally require a valid API key
        # We're just testing that the factory is callable
        assert callable(factory)


class TestProviderRegistryIsRegistered:
    """Test ProviderRegistry.is_registered method."""

    def test_is_registered_true(self, populated_registry: ProviderRegistry) -> None:
        """Test is_registered returns True for registered providers."""
        assert populated_registry.is_registered("openai")
        assert populated_registry.is_registered("anthropic")
        assert populated_registry.is_registered("google")

    def test_is_registered_false(self, empty_registry: ProviderRegistry) -> None:
        """Test is_registered returns False for unregistered providers."""
        assert not empty_registry.is_registered("openai")
        assert not empty_registry.is_registered("nonexistent")

    def test_is_registered_case_sensitive(
        self, populated_registry: ProviderRegistry
    ) -> None:
        """Test is_registered is case-sensitive."""
        assert populated_registry.is_registered("openai")
        assert not populated_registry.is_registered("OpenAI")
        assert not populated_registry.is_registered("OPENAI")


class TestProviderRegistryListProviders:
    """Test ProviderRegistry.list_providers method."""

    def test_list_providers_empty(self, empty_registry: ProviderRegistry) -> None:
        """Test list_providers returns empty list for empty registry."""
        assert empty_registry.list_providers() == []

    def test_list_providers_sorted(self, populated_registry: ProviderRegistry) -> None:
        """Test list_providers returns sorted list."""
        providers = populated_registry.list_providers()
        assert providers == ["anthropic", "google", "openai"]
        assert providers == sorted(providers)

    def test_list_providers_updates_after_register(
        self, empty_registry: ProviderRegistry
    ) -> None:
        """Test list_providers reflects new registrations."""
        assert len(empty_registry.list_providers()) == 0

        empty_registry.register("openai", GPTModel)
        assert len(empty_registry.list_providers()) == 1

        empty_registry.register("anthropic", AnthropicModel)
        assert len(empty_registry.list_providers()) == 2


class TestProviderRegistryUnregister:
    """Test ProviderRegistry.unregister method."""

    def test_unregister_success(self, populated_registry: ProviderRegistry) -> None:
        """Test unregistering a provider."""
        assert populated_registry.is_registered("openai")

        populated_registry.unregister("openai")

        assert not populated_registry.is_registered("openai")
        assert "openai" not in populated_registry.list_providers()

    def test_unregister_not_found(self, empty_registry: ProviderRegistry) -> None:
        """Test unregistering unregistered provider raises ValueError."""
        with pytest.raises(ValueError, match="not registered"):
            empty_registry.unregister("nonexistent")

    def test_unregister_then_register(
        self, populated_registry: ProviderRegistry
    ) -> None:
        """Test can re-register after unregistering."""
        populated_registry.unregister("openai")
        assert not populated_registry.is_registered("openai")

        # Should not raise error since it's unregistered
        populated_registry.register("openai", MockLLMProvider)
        assert populated_registry.is_registered("openai")
        assert populated_registry.get_factory("openai") == MockLLMProvider


class TestProviderRegistryClear:
    """Test ProviderRegistry.clear method."""

    def test_clear_removes_all(self, populated_registry: ProviderRegistry) -> None:
        """Test clear removes all providers."""
        assert len(populated_registry.list_providers()) == 3

        populated_registry.clear()

        assert len(populated_registry.list_providers()) == 0
        assert not populated_registry.is_registered("openai")
        assert not populated_registry.is_registered("anthropic")
        assert not populated_registry.is_registered("google")

    def test_clear_empty_registry(self, empty_registry: ProviderRegistry) -> None:
        """Test clear on empty registry doesn't raise error."""
        empty_registry.clear()  # Should not raise
        assert len(empty_registry.list_providers()) == 0


class TestProviderRegistryIntegration:
    """Integration tests for ProviderRegistry."""

    def test_register_and_use_custom_provider(
        self, empty_registry: ProviderRegistry
    ) -> None:
        """Test end-to-end registration and usage of custom provider."""
        # Register custom provider
        empty_registry.register("mock-provider", MockLLMProvider)

        # Get factory
        factory = empty_registry.get_factory("mock-provider")

        # Instantiate client
        client = factory(model="mock-model-1", api_key="test-key")

        # Verify it works
        assert isinstance(client, MockLLMProvider)
        assert client.model == "mock-model-1"
        assert client.api_key == "test-key"
        assert client.get_model_name() == "mock-model-1"

    def test_default_registry_isolation(self) -> None:
        """Test that default_registry is isolated from new registries."""
        custom_registry = ProviderRegistry()
        custom_registry.register("custom", MockLLMProvider)

        # Custom provider should only be in custom_registry
        assert custom_registry.is_registered("custom")
        assert not default_registry.is_registered("custom")

        # Default providers should only be in default_registry
        assert default_registry.is_registered("openai")
        assert not custom_registry.is_registered("openai")

    def test_multiple_registries_independent(self) -> None:
        """Test that multiple registries are independent."""
        registry1 = ProviderRegistry()
        registry2 = ProviderRegistry()

        registry1.register("openai", GPTModel)
        registry2.register("anthropic", AnthropicModel)

        assert registry1.is_registered("openai")
        assert not registry1.is_registered("anthropic")

        assert registry2.is_registered("anthropic")
        assert not registry2.is_registered("openai")

    def test_workflow_with_overrides(self, empty_registry: ProviderRegistry) -> None:
        """Test realistic workflow with registration and overrides."""
        # Start with standard provider
        empty_registry.register("openai", GPTModel)
        factory1 = empty_registry.get_factory("openai")
        assert factory1 == GPTModel

        # Override with custom implementation
        empty_registry.register("openai", MockLLMProvider, force=True)
        factory2 = empty_registry.get_factory("openai")
        assert factory2 == MockLLMProvider

        # Unregister and re-register original
        empty_registry.unregister("openai")
        empty_registry.register("openai", GPTModel)
        factory3 = empty_registry.get_factory("openai")
        assert factory3 == GPTModel


class TestDefaultRegistryProviders:
    """Test that default_registry has all expected providers correctly configured."""

    @pytest.mark.parametrize(
        "provider,expected_factory",
        [
            ("openai", GPTModel),
            ("anthropic", AnthropicModel),
            ("google", GeminiModel),
            ("ollama", OllamaModel),
        ],
    )
    def test_default_provider_factories(
        self, provider: str, expected_factory: type[DeepEvalBaseLLM]
    ) -> None:
        """Test that default providers map to correct factories."""
        assert default_registry.is_registered(provider)
        assert default_registry.get_factory(provider) == expected_factory

    def test_all_default_providers_instantiable(self) -> None:
        """Test that all default provider factories are valid classes."""
        for provider in default_registry.list_providers():
            factory = default_registry.get_factory(provider)

            # Verify it's a class
            assert isinstance(factory, type)

            # Verify it's a subclass of DeepEvalBaseLLM
            assert issubclass(factory, DeepEvalBaseLLM)
