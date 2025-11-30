"""Type definitions for the adaptive router application.

This module contains all Pydantic models and type definitions used by the
application layer. No business logic should be in this file - only type definitions.

These types match the Go structs in adaptive-model-registry/internal/models/model.go
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# Exception classes
class RegistryError(Exception):
    """Base error type for registry client failures."""


class RegistryConnectionError(RegistryError):
    """Raised when the registry cannot be reached."""


class RegistryResponseError(RegistryError):
    """Raised when the registry returns an unexpected response."""


# Configuration model
class RegistryClientConfig(BaseModel):
    """Configuration for the registry client.

    Attributes:
        base_url: Base URL of the registry service
        timeout: Request timeout in seconds
        default_headers: Optional default headers for requests
    """

    model_config = ConfigDict(frozen=True)

    base_url: str
    timeout: float = 5.0
    default_headers: dict[str, str] | None = None

    def normalized_headers(self) -> dict[str, str]:
        """Return normalized headers dictionary."""
        return dict(self.default_headers or {})


# Registry model
class RegistryModel(BaseModel):
    """Pydantic representation of the normalized Go registry Model struct.

    Attributes:
        id: Database ID
        author: Model author (e.g., "openai", "anthropic")
        model_name: Model name
        display_name: Human-readable display name
        description: Model description
        context_length: Maximum context window size
        pricing: Normalized pricing information (ModelPricing entity)
        architecture: Normalized architecture details (ModelArchitecture entity)
        top_provider: Top provider information (ModelTopProvider entity)
        supported_parameters: List of ModelSupportedParameter entities
        default_parameters: ModelDefaultParameters entity with JSONB parameters
        providers: List of ModelEndpoint entities with nested pricing
        created_at: Creation timestamp
        last_updated: Last update timestamp
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: int | None = None
    author: str  # REQUIRED
    model_name: str  # REQUIRED
    display_name: str | None = Field(default=None, alias="display_name")  # OPTIONAL
    description: str | None = None  # OPTIONAL
    context_length: int | None = Field(default=None, alias="context_length")  # OPTIONAL
    pricing: PricingModel | None = None
    architecture: ArchitectureModel | None = None
    top_provider: TopProviderModel | None = Field(default=None, alias="top_provider")
    supported_parameters: list[SupportedParameterModel] | None = Field(
        default=None, alias="supported_parameters"
    )
    default_parameters: dict[str, Any] | None = Field(
        default=None, alias="default_parameters"
    )
    providers: list[EndpointModel] | None = None
    created_at: datetime | None = Field(default=None, alias="created_at")
    last_updated: datetime | None = Field(default=None, alias="last_updated")

    def unique_id(self) -> str:
        """Construct the router-compatible unique identifier.

        Returns:
            Unique identifier in format "author/model_name"

        Raises:
            RegistryError: If author or model_name is missing
        """
        author = (self.author or "").strip().lower()
        if not author:
            raise RegistryError("registry model missing author field")

        if not self.model_name:
            raise RegistryError(f"registry model '{author}' missing model_name")

        model_name = self.model_name.strip().lower()

        return f"{author}/{model_name}"

    def average_price(self) -> float | None:
        """Calculate average price from available pricing fields.

        Pricing format from normalized registry (ModelPricing entity):
        {
            "prompt_cost": "0.000015",
            "completion_cost": "0.00012",
            "request_cost": "0",
            "image_cost": "0",
            ...
        }

        Returns:
            Average of prompt and completion costs, or None if pricing unavailable
        """
        if not self.pricing:
            return None

        try:
            # Updated field names for normalized schema
            prompt_cost = float(self.pricing.prompt_cost or 0)
            completion_cost = float(self.pricing.completion_cost or 0)

            if prompt_cost == 0 and completion_cost == 0:
                return 0.0

            return (prompt_cost + completion_cost) / 2.0

        except (ValueError, TypeError):
            return None


# ============================================================================
# Adaptive Registry Types (matching Go structs)
# ============================================================================


class PricingModel(BaseModel):
    """Pricing structure for model usage (matches Go Pricing struct)."""

    id: int | None = None
    model_id: int | None = None
    prompt_cost: str | None = None  # Cost per token for input
    completion_cost: str | None = None  # Cost per token for output
    request_cost: str | None = None  # Cost per request
    image_cost: str | None = None  # Cost per image
    web_search_cost: str | None = None  # Cost for web search
    internal_reasoning_cost: str | None = None  # Cost for reasoning


class ArchitectureModalityModel(BaseModel):
    """Architecture modality configuration (matches Go ModelArchitectureModality struct)."""

    id: int | None = None
    architecture_id: int | None = None
    modality_type: str  # REQUIRED - "input" or "output"
    modality_value: str  # REQUIRED - e.g., "text", "image"


class ArchitectureModel(BaseModel):
    """Model architecture and capabilities (matches Go Architecture struct)."""

    id: int | None = None
    model_id: int | None = None
    modality: str  # REQUIRED - e.g., "text+image->text"
    tokenizer: str  # REQUIRED - e.g., "GPT", "Llama3", "Nova"
    instruct_type: str | None = None  # OPTIONAL - e.g., "chatml"
    modalities: list[ArchitectureModalityModel] | None = (
        None  # OPTIONAL - Nested modality relationships
    )


class TopProviderModel(BaseModel):
    """Top provider configuration (matches Go TopProvider struct)."""

    id: int | None = None
    model_id: int | None = None
    context_length: int | None = None
    max_completion_tokens: int | None = None
    is_moderated: str | None = None  # stored as string "true"/"false" in database


class SupportedParameterModel(BaseModel):
    """Supported parameter configuration (matches Go ModelSupportedParameter struct)."""

    id: int | None = None
    model_id: int | None = None
    parameter_name: str  # REQUIRED


class EndpointModel(BaseModel):
    """Provider endpoint configuration (matches Go Endpoint struct)."""

    id: int | None = None
    model_id: int | None = None
    name: str  # REQUIRED
    endpoint_model_name: str  # REQUIRED
    context_length: int  # REQUIRED
    provider_name: str  # REQUIRED
    tag: str  # REQUIRED
    status: int  # REQUIRED
    quantization: str | None = None  # OPTIONAL
    max_completion_tokens: int | None = None  # OPTIONAL
    max_prompt_tokens: int | None = None  # OPTIONAL
    uptime_last_30m: str | None = None  # OPTIONAL - stored as string in database
    supports_implicit_caching: str | None = (
        None  # OPTIONAL - stored as string "true"/"false" in database
    )
    pricing: PricingModel | None = None  # OPTIONAL - ModelEndpointPricing relationship
    supported_parameters: list[SupportedParameterModel] | None = None  # OPTIONAL


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
