"""Public API models for model selection requests and responses.

This module contains the public-facing API models that external users interact with
when making model selection requests to the adaptive router service.
"""

import math

from pydantic import BaseModel, Field, field_validator


class Model(BaseModel):
    """Model specification for routing with cost information.

    Contains the essential fields needed for model identification
    and routing decisions, including mandatory cost data and per-cluster error rates.

    Attributes:
        provider: Model provider (e.g., "openai", "anthropic")
        model_name: Model name (e.g., "gpt-4", "claude-sonnet-4-5")
        cost_per_1m_input_tokens: Cost per 1M input tokens
        cost_per_1m_output_tokens: Cost per 1M output tokens
        error_rates: Per-cluster error rates (K values, one per cluster)
    """

    provider: str
    model_name: str
    cost_per_1m_input_tokens: float = Field(
        ..., ge=0, description="Cost per 1M input tokens"
    )
    cost_per_1m_output_tokens: float = Field(
        ..., ge=0, description="Cost per 1M output tokens"
    )
    error_rates: list[float] = Field(
        default_factory=list, description="Per-cluster error rates (K values)"
    )

    @field_validator("error_rates")
    @classmethod
    def validate_error_rates(cls, v: list[float]) -> list[float]:
        """Validate error rates are finite and in [0.0, 1.0] range.

        Args:
            v: List of error rates to validate

        Returns:
            Validated error rates

        Raises:
            ValueError: If any error rate is invalid
        """
        for i, rate in enumerate(v):
            if not isinstance(rate, (int, float)):
                raise ValueError(
                    f"Invalid error_rate[{i}]: must be a number, got {type(rate).__name__}"
                )
            if not math.isfinite(rate):
                raise ValueError(f"Invalid error_rate[{i}]: {rate} (must be finite)")
            if not (0.0 <= rate <= 1.0):
                raise ValueError(
                    f"Invalid error_rate[{i}]: {rate} (must be in range [0.0, 1.0])"
                )
        return v

    @property
    def cost_per_1m_tokens(self) -> float:
        """Average cost per million tokens (for routing calculations)."""
        return (self.cost_per_1m_input_tokens + self.cost_per_1m_output_tokens) / 2.0

    def unique_id(self) -> str:
        """Construct the router-compatible unique identifier.

        Returns:
            Unique identifier in format "author/model_name"

        Raises:
            ValueError: If author or model_name is empty
        """
        provider = (self.provider or "").strip().lower()
        if not provider:
            raise ValueError("Model missing provider field")

        model_name = self.model_name.strip().lower()
        if not model_name:
            raise ValueError(f"Model '{provider}' missing model_name")

        return f"{provider}/{model_name}"


class ModelSelectionRequest(BaseModel):
    """Model selection request for intelligent routing.

    Contains the prompt and context information needed for intelligent model
    routing, including tool usage detection and user preferences.

    Attributes:
        prompt: The user prompt to analyze
        tool_call: Current tool call being made (for function calling detection)
        tools: Available tool definitions
        user_id: User identifier for tracking
        models: Optional list of model IDs (strings) to restrict routing to
        cost_bias: Cost preference (0.0=cheap, 1.0=quality)
        complexity_threshold: Complexity threshold for model selection
        token_threshold: Token count threshold for model selection
    """

    prompt: str = Field(..., min_length=1)
    user_id: str | None = None
    models: list[str] | None = None
    cost_bias: float | None = None

    @field_validator("cost_bias")
    @classmethod
    def validate_cost_bias(cls, v: float | None) -> float | None:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Cost bias must be between 0.0 and 1.0")
        return v


class Alternative(BaseModel):
    """Alternative model option for routing.

    Attributes:
        model_id: Model identifier (e.g., "anthropic:claude-sonnet-4-5")
    """

    model_id: str = Field(..., min_length=1)


class ModelSelectionResponse(BaseModel):
    """Clean response with selected model and alternatives.

    Attributes:
        model_id: Selected model identifier (e.g., "anthropic:claude-sonnet-4-5")
        alternatives: List of alternative model options
    """

    model_id: str = Field(..., min_length=1)
    alternatives: list[Alternative]
