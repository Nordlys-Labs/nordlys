"""Intelligent model routing using cluster-based selection.

This module provides the ModelRouter class for selecting optimal LLM models
based on cluster-specific error rates, cost optimization, and model capabilities.

All routing logic is consolidated here: MinIO loading, cluster-based routing,
and response conversion.
"""

from __future__ import annotations

import heapq
import logging
import re
import time
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from adaptive_router.loaders.local import LocalFileProfileLoader
from adaptive_router.loaders.minio import MinIOProfileLoader
from adaptive_router.models.api import (
    Alternative,
    Model,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_router.models.routing import (
    ModelFeatureVector,
    ModelScore,
    AlternativeScore,
)
from adaptive_router.models.storage import RouterProfile, MinIOSettings
from adaptive_router.core.cluster_engine import ClusterEngine
from adaptive_router.exceptions.core import (
    ModelNotFoundError,
    InvalidModelFormatError,
)


logger = logging.getLogger(__name__)

_EPSILON = 1e-10


class ModelRouter:
    """Intelligent model routing using cluster-based selection.

    Selects optimal LLM models based on cluster-specific error rates,
    cost optimization, and model capability matching.

    Loads cluster profiles from MinIO S3 storage and performs intelligent
    routing using the UniRouter algorithm with per-cluster error rates.
    """

    def __init__(
        self,
        profile: str | Path | dict[str, Any] | RouterProfile,
    ) -> None:
        """Initialize ModelRouter from profile.

        Args:
            profile: Router profile as:
                - str/Path: Local file path or S3 URL (s3://bucket/key)
                - dict: Profile dictionary
                - RouterProfile: Profile object
                Models are loaded from the profile automatically.
            lambda_min: Minimum lambda value for cost-quality tradeoff (default: 0.0)
            lambda_max: Maximum lambda value for cost-quality tradeoff (default: 2.0)
            default_cost_preference: Default cost preference when not specified (default: 0.5)
            allow_trust_remote_code: Allow execution of remote code in embedding models.
                                   WARNING: Enabling this allows arbitrary code execution from
                                   remote sources and should only be used with trusted models.
                                   Defaults to False for security.

        Raises:
            ModelNotFoundError: If model validation fails or profile models are invalid
        """
        # Load profile internally
        profile = self._load_profile(profile)

        # Get models from profile
        models = profile.models

        # Validate model IDs
        self._validate_model_ids(models)

        n_clusters = profile.metadata.n_clusters
        logger.info(
            f"Initializing ModelRouter with {n_clusters} clusters and {len(models)} models"
        )

        allow_trust_remote_code = profile.metadata.allow_trust_remote_code

        if allow_trust_remote_code:
            logger.warning(
                "WARNING: allow_trust_remote_code=True enables execution of remote code "
                "from embedding models. This should only be used with trusted models."
            )

        self.cluster_engine = self._build_cluster_engine_from_data(profile)

        # Read routing parameters from profile with sensible defaults
        self.lambda_min = profile.metadata.routing.lambda_min
        self.lambda_max = profile.metadata.routing.lambda_max
        self.default_cost_preference = profile.metadata.routing.default_cost_preference

        logger.info(
            f"Loaded cluster engine: {n_clusters} clusters, "
            f"silhouette score: {profile.metadata.silhouette_score or 'N/A'}"
        )

        # Direct iteration over models - no dict lookups needed
        self.model_features: dict[str, ModelFeatureVector] = {}

        for model in profile.models:
            model_id = model.unique_id()

            self.model_features[model_id] = ModelFeatureVector(
                error_rates=model.error_rates,
                cost_per_1m_input_tokens=model.cost_per_1m_input_tokens,
                cost_per_1m_output_tokens=model.cost_per_1m_output_tokens,
            )
            logger.debug(f"Loaded profile for {model_id}")

        if not self.model_features:
            raise ModelNotFoundError("No valid model features found in profile")

        all_costs = [f.cost_per_1m_tokens for f in self.model_features.values()]
        self.min_cost = min(all_costs)
        self.max_cost = max(all_costs)

    @cached_property
    def cost_range(self) -> float:
        """Cache cost range calculation for performance."""
        return self.max_cost - self.min_cost

    def _resolve_cost_preference(
        self, cost_bias: float | None, request_cost_bias: float | None
    ) -> float:
        """Resolve cost preference from multiple sources."""
        if cost_bias is not None:
            return cost_bias
        if request_cost_bias is not None:
            return request_cost_bias
        return self.default_cost_preference

    def _get_models_to_score(
        self, allowed_model_ids: list[str] | None
    ) -> dict[str, ModelFeatureVector]:
        """Get models to score, filtered if allowed_model_ids provided."""
        if allowed_model_ids is not None:
            allowed_set = set(allowed_model_ids)
            models_to_score = {
                model_id: features
                for model_id, features in self.model_features.items()
                if model_id in allowed_set
            }

            if not models_to_score:
                raise ModelNotFoundError(
                    f"No valid models found in allowed list. "
                    f"Allowed: {allowed_model_ids}, Available: {list(self.model_features.keys())}"
                )
        else:
            models_to_score = self.model_features

        return models_to_score

    def _score_models(
        self,
        models_to_score: dict[str, ModelFeatureVector],
        cluster_id: int,
        lambda_param: float,
    ) -> list[tuple[float, str, ModelScore]]:
        """Score all models for the given cluster."""
        scored_models: list[tuple[float, str, ModelScore]] = []

        for model_id, features in models_to_score.items():
            error_rate = features.error_rates[cluster_id]
            cost = features.cost_per_1m_tokens
            normalized_cost = self._normalize_cost(cost)
            score = error_rate + lambda_param * normalized_cost

            heapq.heappush(
                scored_models,
                (
                    score,
                    model_id,
                    ModelScore(
                        score=score,
                        error_rate=error_rate,
                        accuracy=1.0 - error_rate,
                        cost=cost,
                        normalized_cost=normalized_cost,
                    ),
                ),
            )

        return heapq.nsmallest(len(scored_models), scored_models, key=lambda x: x[0])

    def _build_response(
        self, best_model_id: str, top_models: list[tuple[float, str, ModelScore]]
    ) -> ModelSelectionResponse:
        """Build response from scored models."""
        alternatives: list[AlternativeScore] = [
            AlternativeScore(
                model_id=model_id,
                score=model_score.score,
                accuracy=model_score.accuracy,
                cost=model_score.cost,
            )
            for _, model_id, model_score in top_models[1:]
        ]

        alternatives_list = [Alternative(model_id=alt.model_id) for alt in alternatives]

        return ModelSelectionResponse(
            model_id=best_model_id,
            alternatives=alternatives_list,
        )

    def _validate_model_ids(self, models: list[Model]) -> None:
        """Validate model IDs during initialization.

        Args:
            models: List of models to validate

        Raises:
            InvalidModelFormatError: If any model ID is invalid
        """
        pattern = re.compile(r"^[\w-]+/[\w.-]+$")

        for model in models:
            model_id = model.unique_id()
            if not pattern.match(model_id):
                raise InvalidModelFormatError(
                    f"Invalid model ID format: {model_id}. "
                    f"Expected format: 'provider/model_name' (e.g., 'openai/gpt-4')"
                )

    def _load_profile(
        self, profile: str | Path | dict[str, Any] | RouterProfile
    ) -> RouterProfile:
        """Load profile from various sources.

        Args:
            profile: Profile as path (str/Path), dict, or RouterProfile object

        Returns:
            Loaded RouterProfile

        Raises:
            FileNotFoundError: If profile file doesn't exist
            ValueError: If profile format is invalid
        """
        # If already a RouterProfile, return it
        if isinstance(profile, RouterProfile):
            logger.debug("Profile already loaded")
            return profile

        # If dict, parse as RouterProfile
        if isinstance(profile, dict):
            logger.debug("Loading profile from dict")
            return RouterProfile(**profile)

        # If str or Path, load from local file
        # (S3/MinIO profiles should be loaded before passing to router)
        logger.info(f"Loading profile from local file: {profile}")
        loader = LocalFileProfileLoader(str(profile))
        return loader.load_profile()

    def _build_cluster_engine_from_data(
        self,
        profile: RouterProfile,
    ) -> ClusterEngine:
        """Build ClusterEngine from storage profile data.

        Args:
            profile: Validated RouterProfile from storage
            allow_trust_remote_code: Whether to allow remote code execution

        Returns:
            Reconstructed ClusterEngine
        """

        allow_trust_remote_code = profile.metadata.allow_trust_remote_code
        # Load and configure embedding model
        embedding_model = self._load_embedding_model(
            profile.metadata.embedding_model, allow_trust_remote_code
        )

        # Restore cluster engine with K-means parameters
        cluster_engine = self._restore_cluster_engine(profile, embedding_model)

        return cluster_engine

    def _load_embedding_model(
        self,
        embedding_model_name: str,
        allow_trust_remote_code: bool,
    ) -> SentenceTransformer:
        """Load and configure the embedding model.

        Args:
            embedding_model_name: Name of the SentenceTransformer model
            allow_trust_remote_code: Whether to allow remote code execution

        Returns:
            Configured SentenceTransformer model
        """

        device = ClusterEngine.get_device()
        logger.info(f"Loading embedding model on device: {device}")

        # Suppress the clean_up_tokenization_spaces warning during model loading
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
        # Explicitly set clean_up_tokenization_spaces to False for future compatibility
        embedding_model.tokenizer.clean_up_tokenization_spaces = False

        return embedding_model

    def _restore_cluster_engine(
        self,
        profile: RouterProfile,
        embedding_model: SentenceTransformer,
    ) -> ClusterEngine:
        """Restore ClusterEngine from profile data.

        Args:
            profile: RouterProfile with cluster data
            embedding_model: Loaded SentenceTransformer model

        Returns:
            Configured ClusterEngine
        """
        cluster_centers = np.array(profile.cluster_centers.cluster_centers)

        cluster_engine = ClusterEngine.from_fitted_state(
            cluster_centers=cluster_centers,
            n_clusters=profile.cluster_centers.n_clusters,
            embedding_model=embedding_model,
            embedding_model_name=profile.metadata.embedding_model,
            silhouette_score=profile.metadata.silhouette_score or 0.0,
            clustering_config=profile.metadata.clustering,
        )

        logger.info(
            f"Built cluster engine from storage data: {profile.cluster_centers.n_clusters} clusters, "
            f"{profile.cluster_centers.feature_dim} features"
        )

        return cluster_engine

    def select_model(
        self,
        request: ModelSelectionRequest,
        *,
        cost_bias: float | None = None,
    ) -> ModelSelectionResponse:
        """Select optimal model based on prompt analysis.

        This is the main public API method. Uses cluster-based routing to select
        the best model based on prompt characteristics, cost preferences, and
        historical per-cluster error rates.

        Performance Characteristics:
        - Feature extraction (sentence transformers + TF-IDF): 20-50ms
        - Cluster assignment (K-means predict): <5ms
        - Model scoring (heap operations): <5ms
        - Total latency: 30-60ms end-to-end
        - Throughput: 100-500 requests/second (CPU-bound, depends on core count)

        Args:
            request: ModelSelectionRequest with prompt and optional model constraints
            cost_bias: Override default cost preference (0.0=cheap, 1.0=quality).
                      If not provided, uses the request's cost_bias or default_cost_preference.

        Returns:
            ModelSelectionResponse with selected provider, model, and alternatives

        Raises:
            ModelNotFoundError: If requested models are not supported
            InvalidModelFormatError: If model ID format is invalid

        Examples:
            >>> router.select_model(ModelSelectionRequest(prompt="How do I sort a list?"))
            >>> router.select_model(request, cost_bias=0.8)  # Quality-focused
        """
        start_time = time.time()

        allowed_model_ids: list[str] | None = None
        if request.models is not None:
            allowed_model_ids = self._filter_models_by_request(request.models)

        cost_preference = self._resolve_cost_preference(cost_bias, request.cost_bias)
        cluster_id, _ = self.cluster_engine.assign_single(request.prompt)
        lambda_param = self._calculate_lambda(cost_preference)

        models_to_score = self._get_models_to_score(allowed_model_ids)
        top_models = self._score_models(models_to_score, cluster_id, lambda_param)

        _, best_model_id, best_scores = top_models[0]

        self._generate_reasoning(
            cluster_id=cluster_id,
            cost_preference=cost_preference,
            lambda_param=lambda_param,
            selected_scores=best_scores,
        )

        response = self._build_response(best_model_id, top_models)

        routing_time = (time.time() - start_time) * 1000
        logger.info(
            f"Selected model: {best_model_id} "
            f"(cluster {cluster_id}, "
            f"accuracy {best_scores.accuracy:.2%}, "
            f"score {best_scores.score:.3f}, "
            f"routing_time {routing_time:.2f}ms)"
        )

        return response

    def route(self, prompt: str, cost_bias: float | None = None) -> str:
        """Quick routing - just give me the model ID.

        Convenience method for simple routing use cases. Returns only the selected
        model ID without the full response object.

        Args:
            prompt: Question text to route
            cost_bias: Cost preference (0.0=cheap, 1.0=quality). Uses default if None.

        Returns:
            Model ID string (e.g., "openai/gpt-4")

        Raises:
            ModelNotFoundError: If no suitable models found
            InvalidModelFormatError: If model ID format is invalid

        Examples:
            >>> router.route("How do I sort a list?")
            'openai/gpt-3.5-turbo'
            >>> router.route("Complex analysis needed", cost_bias=0.9)
            'openai/gpt-4'
        """
        request = ModelSelectionRequest(prompt=prompt, cost_bias=cost_bias)
        response = self.select_model(request, cost_bias=cost_bias)
        return response.model_id

    @classmethod
    def from_profile(
        cls,
        profile: RouterProfile,
    ) -> ModelRouter:
        return cls(
            profile=profile,
        )

    @classmethod
    def from_minio(
        cls,
        settings: MinIOSettings,
    ) -> ModelRouter:
        loader = MinIOProfileLoader.from_settings(settings)
        profile = loader.load_profile()
        return cls.from_profile(
            profile=profile,
        )

    @classmethod
    def from_local_file(
        cls,
        profile_path: str | Path,
    ) -> ModelRouter:
        loader = LocalFileProfileLoader(profile_path=Path(profile_path))
        profile = loader.load_profile()
        return cls.from_profile(
            profile=profile,
        )

    def _filter_models_by_request(self, models: list[Model]) -> list[str] | None:
        """Filter supported models based on request model specifications.

        This method provides a scalable filtering mechanism that can handle:
        - Full model specifications (provider + model_name)
        - Provider-only filters (just provider specified)
        - Future filters (e.g., cost thresholds, capabilities, etc.)

        Args:
            models: List of Model objects with filter criteria

        Returns:
            List of allowed model IDs in "provider/model_name" format,
            or None if no filtering should be applied

        Raises:
            ModelNotFoundError: If requested models are not supported or no models match filters
        """
        supported = self.get_supported_models()
        logger.debug(
            "Filtering request models. supported=%d request=%d",
            len(supported),
            len(models),
        )

        # Separate full model specs from partial filters
        explicit_models = []
        provider_filters = []

        for m in models:
            if m.provider and m.model_name:
                # Full model specification - exact match required
                model_id = f"{m.provider.lower()}/{m.model_name.lower()}"
                explicit_models.append(model_id)
            elif m.provider:
                # Provider-only filter - match all models from this provider
                provider_filters.append(m.provider.lower())
            # Future: Add more filter types here (e.g., cost_threshold, supports_function_calling, etc.)

        # Apply filters in order of specificity
        allowed_model_ids = []

        # 1. Explicit model specifications take highest priority
        if explicit_models:
            # Filter to only supported models instead of raising error
            unsupported = [m for m in explicit_models if m not in supported]
            if unsupported:
                logger.warning(
                    f"Dropping unsupported models from request: {unsupported}. "
                    f"Supported models: {supported}"
                )
            # Only add models that are actually supported
            supported_explicit = [m for m in explicit_models if m in supported]
            allowed_model_ids.extend(supported_explicit)

        # 2. Apply provider filters
        if provider_filters:
            provider_filtered = [
                model_id
                for model_id in supported
                if any(
                    model_id.startswith(f"{provider}/") for provider in provider_filters
                )
            ]
            if not provider_filtered:
                raise ModelNotFoundError(
                    f"No supported models found for providers: {provider_filters}. "
                    f"Supported models: {supported}"
                )
            allowed_model_ids.extend(provider_filtered)
            logger.debug(
                "Provider filters matched %d models: %s",
                len(provider_filtered),
                provider_filtered,
            )

        # 3. Future filters can be added here (e.g., capability filters, cost filters)
        # Example:
        # if cost_threshold_filters:
        #     cost_filtered = [
        #         model_id for model_id in supported
        #         if self.model_features[model_id].cost_per_1m_tokens <= threshold
        #     ]
        #     allowed_model_ids.extend(cost_filtered)

        # Remove duplicates while preserving order (using dict.fromkeys)
        if allowed_model_ids:
            unique_models = list(dict.fromkeys(allowed_model_ids))
            if not unique_models:
                raise ModelNotFoundError(
                    "No supported models remain after filtering. "
                    f"Requested models were filtered out. Supported models: {supported}"
                )
            logger.debug("Final allowed models after filtering: %s", unique_models)
            return unique_models

        if models:
            requested_descriptions: list[str] = []
            for m in models:
                if m.provider and m.model_name:
                    requested_descriptions.append(
                        f"{m.provider.lower()}/{m.model_name.lower()}"
                    )
                elif m.provider:
                    requested_descriptions.append(f"{m.provider.lower()}/*")
            requested_text = requested_descriptions or ["<unspecified>"]
            raise ModelNotFoundError(
                "Requested models are not supported by the router: "
                f"{requested_text}. Supported models: {supported}"
            )

        # No filters specified - use all models
        return None

    def _calculate_lambda(self, cost_preference: float) -> float:
        """Calculate lambda parameter.

        Args:
            cost_preference: 0.0=cheap, 1.0=quality

        Returns:
            Lambda parameter (higher = more cost penalty)
        """
        # Invert: high quality preference = low lambda (cost matters less)
        lambda_param = self.lambda_max - cost_preference * (
            self.lambda_max - self.lambda_min
        )

        return lambda_param

    def _normalize_cost(self, cost: float) -> float:
        """Normalize cost to [0, 1] range.

        Args:
            cost: Model cost per 1M tokens

        Returns:
            Normalized cost
        """
        if self.cost_range < _EPSILON:
            return 0.0

        return float((cost - self.min_cost) / self.cost_range)

    def _generate_reasoning(
        self,
        cluster_id: int,
        cost_preference: float,
        lambda_param: float,
        selected_scores: ModelScore,
    ) -> str:
        """Generate human-readable reasoning for routing decision.

        Args:
            cluster_id: Assigned cluster
            cost_preference: User's cost preference
            lambda_param: Calculated lambda
            selected_scores: Scores for selected model

        Returns:
            Reasoning string
        """
        parts = []

        # Cluster info
        parts.append(f"Question assigned to cluster {cluster_id}")

        # Preference info
        if cost_preference < 0.3:
            parts.append(f"Cost-optimized routing (λ={lambda_param:.2f})")
        elif cost_preference < 0.7:
            parts.append(f"Balanced cost-accuracy routing (λ={lambda_param:.2f})")
        else:
            parts.append(f"Quality-optimized routing (λ={lambda_param:.2f})")

        # Performance info
        accuracy = selected_scores.accuracy
        if accuracy >= 0.95:
            parts.append(f"Excellent predicted accuracy ({accuracy:.0%})")
        elif accuracy >= 0.75:
            parts.append(f"Strong predicted accuracy ({accuracy:.0%})")
        else:
            parts.append(f"Best available option ({accuracy:.0%} predicted)")

        return "; ".join(parts)

    def get_supported_models(self) -> list[str]:
        """Get list of models this router supports.

        Returns:
            List of model IDs in format "provider/model_name"
        """
        return list(self.model_features.keys())

    def get_cluster_info(self) -> dict[str, Any]:
        """Get information about loaded clusters.

        Returns:
            Dictionary with cluster statistics including n_clusters,
            embedding_model, supported_models, and lambda parameters
        """
        return {
            "n_clusters": self.cluster_engine.n_clusters,
            "embedding_model": self.cluster_engine.embedding_model_name,
            "supported_models": self.get_supported_models(),
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "default_cost_preference": self.default_cost_preference,
        }
