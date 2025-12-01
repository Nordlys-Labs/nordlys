"""Health check models and utilities."""

import time
from enum import Enum

from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health check status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceHealth(BaseModel):
    """Health status of an individual service."""

    status: HealthStatus
    message: str | None = None
    response_time_ms: float | None = None


class HealthCheckResponse(BaseModel):
    """Complete health check response."""

    status: HealthStatus
    models: ServiceHealth
    router: ServiceHealth
    timestamp: float = Field(default_factory=time.time)
