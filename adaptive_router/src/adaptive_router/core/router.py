"""Intelligent model routing using cluster-based selection.

This module provides the ModelRouter class for selecting optimal LLM models
based on cluster-specific error rates, cost optimization, and model capabilities.

Uses C++ core (adaptive_core_ext) for high-performance cluster assignment and
model scoring, with Python handling embedding computation and profile management.
"""

from __future__ import annotations

import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from adaptive_router.loaders.local import LocalFileProfileLoader
from adaptive_router.loaders.minio import MinIOProfileLoader
from adaptive_router.models.api import (
    Alternative,
    Model,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.storage import RouterProfile, MinIOSettings
from adaptive_router.exceptions.core import (
    InvalidModelFormatError,
)
from adaptive_core_ext import Router as CoreRouter


logger = logging.getLogger(__name__)


def _get_device() -> str:
    """Get the best available device for embedding computation."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class ModelRouter:
    """Intelligent model routing using cluster-based selection.

    Selects optimal LLM models based on cluster-specific error rates,
    cost optimization, and model capability matching.

    Uses C++ core for high-performance cluster assignment and model scoring,
    with Python handling embedding computation via SentenceTransformer.

    Use factory methods to create instances:
        - ModelRouter.from_json_file(path) - Load from JSON file
        - ModelRouter.from_msgpack_file(path) - Load from MessagePack file
        - ModelRouter.from_profile(profile) - Load from RouterProfile object (in-memory)
        - ModelRouter.from_minio(settings) - Load from MinIO/S3
    """

    def __init__(
        self,
        core_router: CoreRouter,
        embedding_model: SentenceTransformer,
        profile_metadata: Any,
    ) -> None:
        """Initialize ModelRouter (use factory methods instead).

        Args:
            core_router: Initialized C++ core router
            embedding_model: SentenceTransformer for computing embeddings
            profile_metadata: Profile metadata for configuration
        """
        self._core_router = core_router
        self._embedding_model = embedding_model
        self._metadata = profile_metadata
        self.default_cost_preference = profile_metadata.routing.default_cost_preference

    @classmethod
    def from_json_file(cls, path: str | Path) -> ModelRouter:
        """Create router from JSON profile file.

        Args:
            path: Path to JSON profile file

        Returns:
            Initialized ModelRouter
        """
        path = Path(path)
        logger.info(f"Loading router from JSON file: {path}")

        # Load profile to get metadata
        loader = LocalFileProfileLoader(str(path))
        profile = loader.load_profile()

        # Validate model IDs
        cls._validate_model_ids_static(profile.models)

        # Initialize C++ core from JSON file
        core_router = CoreRouter.from_json_file(str(path))

        # Load embedding model
        embedding_model = cls._load_embedding_model_static(
            profile.metadata.embedding_model,
            profile.metadata.allow_trust_remote_code,
        )

        logger.info(
            f"ModelRouter initialized: {profile.metadata.n_clusters} clusters, "
            f"{len(profile.models)} models"
        )

        return cls(core_router, embedding_model, profile.metadata)

    @classmethod
    def from_msgpack_file(cls, path: str | Path) -> ModelRouter:
        """Create router from MessagePack binary profile file.

        Args:
            path: Path to MessagePack (.msgpack) profile file

        Returns:
            Initialized ModelRouter
        """
        path = Path(path)
        logger.info(f"Loading router from MessagePack file: {path}")

        # Load profile to get metadata (need to read it ourselves)
        import msgpack

        with open(path, "rb") as f:
            profile_dict = msgpack.unpackb(f.read(), raw=False)
        profile = RouterProfile(**profile_dict)

        # Validate model IDs
        cls._validate_model_ids_static(profile.models)

        # Initialize C++ core from MessagePack file
        core_router = CoreRouter.from_msgpack_file(str(path))

        # Load embedding model
        embedding_model = cls._load_embedding_model_static(
            profile.metadata.embedding_model,
            profile.metadata.allow_trust_remote_code,
        )

        logger.info(
            f"ModelRouter initialized: {profile.metadata.n_clusters} clusters, "
            f"{len(profile.models)} models"
        )

        return cls(core_router, embedding_model, profile.metadata)

    @classmethod
    def from_profile(cls, profile: RouterProfile) -> ModelRouter:
        """Create router from RouterProfile object (in-memory).

        Args:
            profile: RouterProfile object

        Returns:
            Initialized ModelRouter
        """
        logger.info("Loading router from in-memory profile")

        # Validate model IDs
        cls._validate_model_ids_static(profile.models)

        # Initialize C++ core from JSON string (in-memory, no files)
        profile_json = json.dumps(profile.model_dump())
        core_router = CoreRouter.from_json_string(profile_json)

        # Load embedding model
        embedding_model = cls._load_embedding_model_static(
            profile.metadata.embedding_model,
            profile.metadata.allow_trust_remote_code,
        )

        logger.info(
            f"ModelRouter initialized: {profile.metadata.n_clusters} clusters, "
            f"{len(profile.models)} models"
        )

        return cls(core_router, embedding_model, profile.metadata)

    @classmethod
    def from_minio(cls, settings: MinIOSettings) -> ModelRouter:
        """Create router by loading profile from MinIO/S3.

        Args:
            settings: MinIO connection settings

        Returns:
            Initialized ModelRouter
        """
        loader = MinIOProfileLoader.from_settings(settings)
        profile = loader.load_profile()
        return cls.from_profile(profile)

    @staticmethod
    def _validate_model_ids_static(models: list[Model]) -> None:
        """Validate model IDs."""
        pattern = re.compile(r"^[\w-]+/[\w.-]+$")

        for model in models:
            model_id = model.unique_id()
            if not pattern.match(model_id):
                raise InvalidModelFormatError(
                    f"Invalid model ID format: {model_id}. "
                    f"Expected format: 'provider/model_name' (e.g., 'openai/gpt-4')"
                )

    @staticmethod
    def _load_embedding_model_static(
        embedding_model_name: str,
        allow_trust_remote_code: bool,
    ) -> SentenceTransformer:
        """Load and configure the embedding model."""
        device = _get_device()
        logger.info(
            f"Loading embedding model '{embedding_model_name}' on device: {device}"
        )

        if allow_trust_remote_code:
            logger.warning(
                "WARNING: allow_trust_remote_code=True enables execution of remote code "
                "from embedding models. This should only be used with trusted models."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*clean_up_tokenization_spaces.*",
                category=FutureWarning,
            )
            embedding_model = SentenceTransformer(
                embedding_model_name,
                device=device,
                trust_remote_code=allow_trust_remote_code,
            )
        embedding_model.tokenizer.clean_up_tokenization_spaces = False

        return embedding_model

    def _resolve_cost_preference(
        self, cost_bias: float | None, request_cost_bias: float | None
    ) -> float:
        """Resolve cost preference from multiple sources."""
        if cost_bias is not None:
            return cost_bias
        if request_cost_bias is not None:
            return request_cost_bias
        return self.default_cost_preference

    def select_model(
        self,
        request: ModelSelectionRequest,
        *,
        cost_bias: float | None = None,
    ) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis.

        Uses C++ core for high-performance cluster assignment and model scoring.

        Performance Characteristics:
        - Embedding computation (sentence transformers): 10-50ms
        - Cluster assignment + model scoring (C++ core): ~0.15ms
        - Total latency: 10-50ms end-to-end

        Args:
            request: ModelSelectionRequest with prompt and optional model constraints
            cost_bias: Override default cost preference (0.0=cheap, 1.0=quality).

        Returns:
            ModelSelectionResponse with selected model and alternatives

        Raises:
            ModelNotFoundError: If requested models are not supported
        """
        start_time = time.time()

        # 1. Compute embedding using Python SentenceTransformer
        embedding = self._embedding_model.encode(
            request.prompt,
            convert_to_numpy=True,
        )

        # 2. Resolve cost preference
        cost_preference = self._resolve_cost_preference(cost_bias, request.cost_bias)

        # 3. Extract model IDs from request.models if provided
        model_filter: list[str] = []
        if request.models:
            model_filter = [model.unique_id() for model in request.models]

        # 4. Route using C++ core (cluster assignment + model scoring)
        core_response = self._core_router.route(
            embedding, cost_preference, model_filter
        )

        # 5. Build Python response
        alternatives = [Alternative(model_id=m) for m in core_response.alternatives]
        response = ModelSelectionResponse(
            model_id=core_response.selected_model,
            alternatives=alternatives,
        )

        routing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Selected model: {core_response.selected_model} "
            f"(cluster {core_response.cluster_id}, "
            f"distance {core_response.cluster_distance:.4f}, "
            f"routing_time {routing_time:.2f}ms)"
        )

        return response

    def route(self, prompt: str, cost_bias: float | None = None) -> str:
        """Quick routing - just give me the model ID.

        Args:
            prompt: Question text to route
            cost_bias: Cost preference (0.0=cheap, 1.0=quality). Uses default if None.

        Returns:
            Model ID string (e.g., "openai/gpt-4")
        """
        request = ModelSelectionRequest(prompt=prompt, cost_bias=cost_bias)
        response = self.select_model(request, cost_bias=cost_bias)
        return response.model_id

    def get_supported_models(self) -> list[str]:
        """Get list of models this router supports."""
        return self._core_router.get_supported_models()

    def get_cluster_info(self) -> dict[str, Any]:
        """Get information about loaded clusters."""
        return {
            "n_clusters": self._core_router.get_n_clusters(),
            "embedding_dim": self._core_router.get_embedding_dim(),
            "embedding_model": self._metadata.embedding_model,
            "supported_models": self.get_supported_models(),
            "default_cost_preference": self.default_cost_preference,
        }
