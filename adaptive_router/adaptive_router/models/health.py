"""Health check models for API health endpoints.

This module contains models for health check responses.
"""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response model.

    Attributes:
        status: Service health status
        version: Service version
        models_loaded: Number of models loaded
        clusters_loaded: Number of clusters loaded
    """

    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(default="1.0.0", description="Service version")
    models_loaded: int = Field(..., description="Number of models loaded")
    clusters_loaded: int = Field(..., description="Number of clusters loaded")
