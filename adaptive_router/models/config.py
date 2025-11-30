"""Configuration models for models and routing parameters.

This module contains configuration models for model metadata, YAML configuration
parsing, and routing algorithm parameters.
"""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Configuration for a single model.

    Attributes:
        id: Unique model identifier (format: "provider:model_name")
        name: Human-readable model name
        provider: Model provider (e.g., "openai", "anthropic")
        cost_per_1m_input_tokens: Cost per 1M input tokens
        cost_per_1m_output_tokens: Cost per 1M output tokens
        description: Model description
    """

    id: str
    name: str
    provider: str
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    description: str

    @property
    def cost_per_1m_tokens(self) -> float:
        """Average cost per million tokens (for backward compatibility)."""
        return (self.cost_per_1m_input_tokens + self.cost_per_1m_output_tokens) / 2.0
