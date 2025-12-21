"""Lightweight re-exports for Deepeval LLM clients.

This module lets scripts import Deepeval models from the nordlys
namespace (e.g., ``from nordlys.evaluators import GPTModel``) so every
callsite shares the same dependency surface.
"""

from deepeval.models.llms import (
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

__all__ = [
    "AmazonBedrockModel",
    "AnthropicModel",
    "AzureOpenAIModel",
    "DeepSeekModel",
    "GeminiModel",
    "GPTModel",
    "GrokModel",
    "KimiModel",
    "LiteLLMModel",
    "LocalModel",
    "OllamaModel",
]
