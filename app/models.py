"""Type definitions for the adaptive router application.

This module contains all Pydantic models and type definitions used by the
application layer. No business logic should be in this file - only type definitions.
"""

from __future__ import annotations


from pydantic import BaseModel, Field, field_validator


# ============================================================================
# API Response Types
# ============================================================================


class ModelSelectionAPIRequest(BaseModel):
    """API request model that accepts model specifications as strings.

    This is the external API model that accepts "author/model_name" or "author/model_name:variant" strings,
    which are then resolved to Model objects internally.
    """

    prompt: str = Field(..., min_length=1)
    user_id: str | None = None
    models: list[str] | None = Field(
        default=None,
        max_length=50,
        description="Optional list of allowed models (max 50 to prevent DoS)",
    )
    cost_bias: float | None = None

    @field_validator("cost_bias")
    @classmethod
    def validate_cost_bias(cls, v: float | None) -> float | None:
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError("Cost bias must be between 0.0 and 1.0")
        return v


class ModelSelectionAPIResponse(BaseModel):
    """Simplified model selection response with model IDs only.

    Attributes:
        selected_model: Model ID string (author/model_name format)
        alternatives: List of model ID strings for alternative models
    """

    selected_model: str
    alternatives: list[str]
