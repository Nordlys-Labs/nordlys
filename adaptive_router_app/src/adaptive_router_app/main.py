"""FastAPI application for adaptive_router service.

Provides HTTP API endpoints for intelligent model selection using cluster-based routing.
Supports Modal serverless deployment with automatic model caching.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, AsyncIterator

import modal
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from adaptive_router.core.router import ModelRouter
from adaptive_router.models.api import Model, ModelSelectionRequest
from adaptive_router.models.storage import RouterProfile
from adaptive_router.exceptions.core import ModelNotFoundError

from adaptive_router_app.models import ModelSelectionAPIRequest
from adaptive_router_app.config import AppSettings
from adaptive_router_app.health import HealthCheckResponse, HealthStatus, ServiceHealth
from adaptive_router_app.models import ModelSelectionAPIResponse

from adaptive_router_app.utils import resolve_models

log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


def extract_model_ids_from_profile(profile: RouterProfile) -> list[str]:
    """Extract model IDs from a RouterProfile."""
    return [model.unique_id() for model in profile.models]


async def create_model_router(
    profile_path: str = "/data/profile.json",
) -> tuple[ModelRouter, RouterProfile]:
    """Create ModelRouter from profile stored in Modal Volume.

    Args:
        profile_path: Path to the RouterProfile JSON file (default: Modal Volume mount)

    Returns:
        Tuple of (ModelRouter, RouterProfile)
    """
    logger.info("Creating ModelRouter from profile: %s", profile_path)

    from adaptive_router.loaders.local import LocalFileProfileLoader

    loader = LocalFileProfileLoader(profile_path)
    profile = loader.load_profile()
    router = ModelRouter.from_profile(profile)

    logger.info("ModelRouter created successfully")

    return router, profile


class AppState:
    """Application state container."""

    def __init__(self) -> None:
        """Initialize application state."""
        self.settings: AppSettings | None = None
        self.router: ModelRouter | None = None
        self.available_models: list[Model] | None = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application with dependency injection."""
    app_state = AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Lifespan event handler for startup/shutdown."""
        try:
            app_state.settings = AppSettings()

            logger.info("Starting Adaptive Router service")
            logger.info("API Docs available at /docs once server is running")

            logger.info("Loading router profile and initializing services...")
            router, profile = await create_model_router()
            app_state.router = router
            app_state.available_models = list(profile.models)
            logger.info(
                "Loaded profile with %d model IDs",
                len(extract_model_ids_from_profile(profile)),
            )
            logger.info("ModelRouter initialized successfully")

            logger.info("FastAPI application started successfully")
        except Exception as e:
            logger.error("Failed to initialize router: %s", e, exc_info=True)
            raise

        yield

        logger.info("Shutting down Adaptive Router...")

    app = FastAPI(
        title="Adaptive Router",
        description="Intelligent LLM model selection API with cluster-based routing",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    def configure_cors() -> None:
        """Configure CORS middleware based on environment.

        Development: Allows all origins and methods for easier testing
        Production: Restricts to specific origins and methods for security
        """
        settings = app_state.settings or AppSettings()
        origins = settings.origins_list
        is_dev = settings.environment.value == "development"

        # Security: Don't allow ["*"] with credentials
        allow_credentials = origins != ["*"]

        # Dev: allow all methods; Prod: restrict to necessary methods
        allowed_methods = ["*"] if is_dev else ["GET", "POST"]

        # Dev: allow all headers; Prod: restrict to necessary headers
        allowed_headers = (
            ["*"]
            if is_dev
            else [
                "Content-Type",
                "Accept",
                "Authorization",
            ]
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=allow_credentials,
            allow_methods=allowed_methods,
            allow_headers=allowed_headers,
        )

    configure_cors()

    @lru_cache()
    def get_settings() -> AppSettings:
        """Get application settings dependency."""
        if app_state.settings is None:
            app_state.settings = AppSettings()
        return app_state.settings

    async def get_router() -> ModelRouter:
        """Get ModelRouter dependency."""
        if app_state.router is None:
            logger.info("Lazy-initializing ModelRouter...")
            router, profile = await create_model_router()
            app_state.router = router
            app_state.available_models = list(profile.models)
            logger.info("ModelRouter initialized successfully")
        return app_state.router

    async def get_available_models() -> list[Model]:
        """Return the list of models from the loaded router profile."""
        if app_state.available_models is None:
            await get_router()
        if app_state.available_models is None:
            raise RuntimeError("Router profile not loaded yet")
        return app_state.available_models

    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        status_code=status.HTTP_200_OK,
        responses={
            200: {"description": "All services healthy"},
            503: {"description": "One or more services unhealthy"},
        },
    )
    async def health_check() -> HealthCheckResponse:
        """Comprehensive health check for router and profile models."""
        overall_status = HealthStatus.HEALTHY

        if app_state.available_models:
            models_health = ServiceHealth(
                status=HealthStatus.HEALTHY,
                message=f"{len(app_state.available_models)} models loaded from profile",
            )
        else:
            overall_status = HealthStatus.UNHEALTHY
            models_health = ServiceHealth(
                status=HealthStatus.UNHEALTHY,
                message="Router profile not loaded",
            )

        if app_state.router is not None:
            router_health = ServiceHealth(
                status=HealthStatus.HEALTHY,
                message="Router initialized",
            )
        else:
            overall_status = (
                HealthStatus.DEGRADED
                if overall_status == HealthStatus.HEALTHY
                else overall_status
            )
            router_health = ServiceHealth(
                status=HealthStatus.DEGRADED,
                message="Router not yet initialized (will initialize on first request)",
            )

        return HealthCheckResponse(
            status=overall_status,
            models=models_health,
            router=router_health,
        )

    @app.post(
        "/select-model",
        response_model=ModelSelectionAPIResponse,
        status_code=status.HTTP_200_OK,
        responses={
            400: {"description": "Invalid request"},
            500: {"description": "Internal server error"},
        },
    )
    async def select_model(
        request: ModelSelectionAPIRequest,
        http_request: Request,
        router: Annotated[ModelRouter, Depends(get_router)],
        available_models: Annotated[list[Model], Depends(get_available_models)],
    ) -> ModelSelectionAPIResponse:
        """Select optimal model based on prompt analysis."""
        start_time = time.perf_counter()

        try:
            resolved_models = None
            if request.models:
                logger.debug("Requested model filters: %s", request.models)
                try:
                    resolved_models = resolve_models(request.models, available_models)
                    logger.debug(
                        "Resolved %d models from request: %s",
                        len(resolved_models),
                        [model.unique_id() for model in resolved_models],
                    )
                except ValueError as e:
                    logger.error("Model resolution failed: %s", e, exc_info=True)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Unable to resolve requested models. Please check your model selections.",
                    ) from e

            internal_request = ModelSelectionRequest(
                prompt=request.prompt,
                user_id=request.user_id,
                models=resolved_models,
                cost_bias=request.cost_bias,
            )

            logger.info(
                "Processing model selection request",
                extra={
                    "prompt_length": len(internal_request.prompt),
                    "cost_bias": internal_request.cost_bias,
                    "models_count": (
                        len(internal_request.models) if internal_request.models else 0
                    ),
                    "client_ip": (
                        http_request.client.host if http_request.client else "unknown"
                    ),
                },
            )

            response = router.select_model(internal_request)
            elapsed = time.perf_counter() - start_time

            logger.info(
                "Model selection completed",
                extra={
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "selected_model_id": response.model_id,
                    "alternatives_count": len(response.alternatives),
                },
            )

            selected_model_id = response.model_id
            alternative_model_ids = [alt.model_id for alt in response.alternatives]

            return ModelSelectionAPIResponse(
                selected_model=selected_model_id,
                alternatives=alternative_model_ids,
            )

        except ModelNotFoundError as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Requested models unavailable: %s",
                e,
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No suitable models available for this request.",
            ) from e

        except ValueError as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Validation error: %s",
                e,
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request parameters.",
            ) from e

        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Model selection failed: %s",
                e,
                extra={"elapsed_ms": round(elapsed * 1000, 2)},
                exc_info=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            ) from e

    return app


# ============================================================================
# Modal Serverless Deployment
# ============================================================================
# Deploy with: modal deploy adaptive_router_app/main.py
# Modal secrets and volumes are configured below.
# Ensure the profile.json file is available in the adaptive-router-data volume.
app = modal.App("adaptive-router")

model_cache = modal.Volume.from_name(
    "sentence-transformer-cache", create_if_missing=True
)

profile_data = modal.Volume.from_name("adaptive-router-data", create_if_missing=True)

# Custom image with dependencies
# NOTE: Modal's uv_sync does NOT support UV workspaces, so we:
# 1. Copy local sources for the cores, library, and app
# 2. pip install the cores (CPU + CUDA), then the library with CUDA extra, then the app
image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "SENTENCE_TRANSFORMERS_HOME": "/vol/model_cache",
        }
    )
    .add_local_python_source(
        "adaptive_router_core",
        copy=True,
        ignore=["build/**", "**/*.so", "**/*.pyd"],
    )
    .add_local_python_source(
        "adaptive_router_core_cu12",
        copy=True,
        ignore=["build/**", "**/*.so", "**/*.pyd"],
    )
    .add_local_python_source("adaptive_router", copy=True)
    .add_local_python_source("adaptive_router_app", copy=True)  # Add app package
    .pip_install(
        [
            "/root/adaptive_router_core",
            "/root/adaptive_router_core_cu12",
            "/root/adaptive_router[cuda]",
            "/root/adaptive_router_app",
        ]
    )
)


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("adaptive-router-secrets")],
    gpu="L40S",
    scaledown_window=600,
    min_containers=0,
    volumes={
        "/vol/model_cache": model_cache,
        "/data": profile_data,
    },
)
@modal.concurrent(max_inputs=100)
class AdaptiveRouterService:
    """Modal service class for Adaptive Router serverless deployment.

    Scales to 0 containers when idle (min_containers=0) for cost efficiency.
    Uses T4 GPU (16GB VRAM) for efficient embedding model inference.
    Scaledown window is 60 seconds before shutdown.
    Supports up to 100 concurrent inputs via @modal.concurrent(max_inputs=100).
    """

    @modal.asgi_app()
    def web_app(self) -> FastAPI:
        """Return FastAPI application for Modal ASGI serving."""
        return create_app()


# Expose FastAPI app for Railway deployment
# Railway will run: hypercorn adaptive_router_app.main:fastapi_app --bind "[::]:$PORT"
fastapi_app = create_app()
