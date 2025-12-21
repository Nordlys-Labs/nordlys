"""Trainer for Adaptive Router profiles.

This module provides the Trainer class for end-to-end training of router profiles,
including clustering, model inference, error rate computation, and profile generation.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from deepeval.dataset import Golden
from deepeval.models import DeepEvalBaseLLM

from nordlys.core.cluster_engine import ClusterEngine
from nordlys.core.provider_registry import default_registry
from nordlys.models.api import Model
from nordlys.models.storage import (
    ClusterCentersData,
    ClusteringConfig,
    ProfileMetadata,
    RouterProfile,
)
from nordlys.models.train import ProviderConfig, TrainingResult


logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for Adaptive Router profiles.

    This class handles end-to-end training workflow:
    1. Data loading from multiple formats
    2. Clustering with ClusterEngine
    3. Optional parallel model inference
    4. Binary correctness evaluation
    5. Profile generation

    Training and saving are separate operations for flexibility.

    Example:
        >>> from nordlys import Trainer, Model, ProviderConfig
        >>> models = [
        ...     Model(provider="openai", model_name="gpt-3.5-turbo", ...),
        ...     Model(provider="openai", model_name="gpt-4", ...),
        ... ]
        >>> provider_configs = {
        ...     "openai": ProviderConfig(api_key="sk-..."),
        ... }
        >>> trainer = Trainer(models=models, provider_configs=provider_configs)
        >>> result = trainer.train_from_csv("data.csv", "question", "answer")
        >>> trainer.save_profile("profile.json")  # Save locally
        >>> trainer.save_profile("s3://bucket/profile.json", s3_settings=config)
    """

    def __init__(
        self,
        models: list[Model],
        provider_configs: dict[str, ProviderConfig],
        n_clusters: int = 20,
        max_parallel: int = 10,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        random_seed: int = 42,
    ):
        """Initialize trainer.

        Args:
            models: List of models to train/profile
            provider_configs: Provider configurations (API keys, base URLs)
                Keys must match model providers
            n_clusters: Number of clusters for K-means
            max_parallel: Maximum parallel inference requests
            embedding_model: Sentence transformer model name
            random_seed: Random seed for reproducibility

        Raises:
            ValueError: If provider configs are missing for any model provider
        """
        self.models = models
        self.provider_configs = provider_configs
        self.n_clusters = n_clusters
        self.max_parallel = max_parallel
        self.embedding_model = embedding_model
        self.random_seed = random_seed

        # Store configurations for profile building
        self.clustering_config = ClusteringConfig(random_state=random_seed)

        # Trained profile (set after training)
        self._trained_profile: RouterProfile | None = None

        # Validate provider configs cover all model providers
        model_providers = {m.provider for m in models}
        missing = model_providers - set(provider_configs.keys())
        if missing:
            raise ValueError(
                f"Missing provider configurations for providers: {missing}. "
                f"Please provide ProviderConfig for each model's provider."
            )

        # Initialize configured ClusterEngine (will be fitted during training)
        self.cluster_engine = ClusterEngine().configure(
            n_clusters=self.n_clusters,
            max_iter=self.clustering_config.max_iter,
            random_state=self.clustering_config.random_state,
            n_init=self.clustering_config.n_init,
            embedding_model=self.embedding_model,
        )

        # Set config on cluster_engine after configure
        self.cluster_engine.clustering_config = self.clustering_config

        logger.info(
            f"Initialized Trainer with {len(models)} models and {n_clusters} clusters"
        )

    def train_from_csv(
        self,
        path: str,
        input_column: str,
        expected_output_column: str,
        actual_output_column: str | None = None,
    ) -> TrainingResult:
        """Train from CSV file.

        Args:
            path: Path to CSV file
            input_column: Column name containing input/question text
            expected_output_column: Column name containing expected/correct output
            actual_output_column: Optional column with pre-computed model outputs
                If None, will run inference on all models

        Returns:
            TrainingResult with training metrics

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            KeyError: If specified columns don't exist in CSV
        """
        logger.info(f"Loading training data from CSV: {path}")
        df = pl.read_csv(path)
        return self.train_from_dataframe(
            df, input_column, expected_output_column, actual_output_column
        )

    def train_from_json(
        self,
        path: str,
        input_column: str,
        expected_output_column: str,
        actual_output_column: str | None = None,
    ) -> TrainingResult:
        """Train from JSON file.

        Args:
            path: Path to JSON file (list of objects or dict of lists)
            input_column: Column name containing input/question text
            expected_output_column: Column name containing expected/correct output
            actual_output_column: Optional column with pre-computed model outputs

        Returns:
            TrainingResult with training metrics

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            KeyError: If specified columns don't exist
        """
        logger.info(f"Loading training data from JSON: {path}")
        with open(path) as f:
            data = json.load(f)
        return self.train_from_dict(
            data, input_column, expected_output_column, actual_output_column
        )

    def train_from_dict(
        self,
        data: list[dict[str, Any]] | dict[str, list[Any]],
        input_column: str,
        expected_output_column: str,
        actual_output_column: str | None = None,
    ) -> TrainingResult:
        """Train from dictionary or list of dicts.

        Args:
            data: Either list of dicts or dict of lists
            input_column: Column name containing input/question text
            expected_output_column: Column name containing expected/correct output
            actual_output_column: Optional column with pre-computed model outputs

        Returns:
            TrainingResult with training metrics

        Raises:
            KeyError: If specified columns don't exist
        """
        logger.info("Loading training data from dict")

        # Normalize to list of dicts
        if isinstance(data, dict):
            # Dict of lists -> list of dicts
            keys = list(data.keys())
            if not keys:
                raise ValueError("Empty data dictionary")

            # Validate all columns have the same length
            lengths = {k: len(data[k]) for k in keys}
            unique_lengths = set(lengths.values())
            if len(unique_lengths) != 1:
                raise ValueError(
                    f"Inconsistent column lengths in data: {lengths}. "
                    f"All columns must have the same number of rows."
                )

            n = len(data[keys[0]])
            data = [{k: data[k][i] for k in keys} for i in range(n)]

        # Extract columns
        try:
            inputs = [str(row[input_column]) for row in data]
            expected = [str(row[expected_output_column]) for row in data]
        except KeyError as e:
            raise KeyError(f"Column {e} not found in data") from e

        # Handle actual outputs
        actuals = None
        if actual_output_column:
            try:
                # Single column -> same for all models
                actual_values = [str(row[actual_output_column]) for row in data]
                actuals = {m.unique_id(): actual_values for m in self.models}
            except KeyError as e:
                raise KeyError(f"Column {e} not found in data") from e

        return self._train(inputs, expected, actuals)

    def train_from_dataframe(
        self,
        df: pl.DataFrame,
        input_column: str,
        expected_output_column: str,
        actual_output_column: str | None = None,
    ) -> TrainingResult:
        """Train from polars DataFrame.

        Args:
            df: Polars DataFrame containing training data
            input_column: Column name containing input/question text
            expected_output_column: Column name containing expected/correct output
            actual_output_column: Optional column with pre-computed model outputs

        Returns:
            TrainingResult with training metrics

        Raises:
            KeyError: If specified columns don't exist in DataFrame
        """
        logger.info(f"Loading training data from DataFrame ({len(df)} rows)")

        try:
            inputs = df[input_column].cast(pl.Utf8).to_list()
            expected = df[expected_output_column].cast(pl.Utf8).to_list()
        except Exception as e:
            raise KeyError(f"Column not found in DataFrame: {e}") from e

        actuals = None
        if actual_output_column:
            try:
                actual_values = df[actual_output_column].cast(pl.Utf8).to_list()
                actuals = {m.unique_id(): actual_values for m in self.models}
            except Exception as e:
                raise KeyError(f"Column not found in DataFrame: {e}") from e

        return self._train(inputs, expected, actuals)

    def train_from_goldens(
        self,
        goldens: list[Golden],
    ) -> TrainingResult:
        """Train from DeepEval Golden objects.

        Args:
            goldens: List of DeepEval Golden objects

        Returns:
            TrainingResult with training metrics
        """
        logger.info(f"Loading training data from {len(goldens)} Golden objects")

        inputs = [g.input for g in goldens]
        expected = [
            str(g.expected_output) if g.expected_output else "" for g in goldens
        ]

        # Check if actual outputs are available
        has_actual = any(g.actual_output for g in goldens)
        actuals = None
        if has_actual:
            actual_values = [
                (str(g.actual_output) if g.actual_output else "") for g in goldens
            ]
            actuals = {m.unique_id(): actual_values for m in self.models}

        return self._train(inputs, expected, actuals)

    def train_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        input_column: str = "input",
        expected_output_column: str = "expected_output",
        actual_output_column: str | None = None,
    ) -> TrainingResult:
        """Train from HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to use (default: "train")
            input_column: Column containing input text
            expected_output_column: Column containing expected output
            actual_output_column: Optional column with pre-computed outputs

        Returns:
            TrainingResult with training metrics

        Raises:
            ImportError: If datasets library not installed
            KeyError: If specified columns don't exist
        """
        from datasets import load_dataset

        logger.info(f"Loading training data from HuggingFace: {dataset_name} ({split})")
        dataset = load_dataset(dataset_name, split=split)
        df = pl.DataFrame(dataset.with_format("arrow"))

        return self.train_from_dataframe(
            df, input_column, expected_output_column, actual_output_column
        )

    def _train(
        self,
        inputs: list[str],
        expected_outputs: list[str],
        actual_outputs: dict[str, list[str]] | None = None,
    ) -> TrainingResult:
        """Core training logic.

        Memory Efficiency:
        - Feature extraction: O(n_samples * feature_dim) = O(n * 5384) for hybrid features
        - Cluster assignments: O(n_samples) for integer array
        - Error rate computation: O(n_clusters * n_models) = O(20 * 5) = O(100) storage
        - Total memory: ~1-2GB for typical training sets (10K samples)
        - For very large datasets (100K+ samples), consider streaming or mini-batch training

        Args:
            inputs: List of input strings
            expected_outputs: List of expected output strings
            actual_outputs: Optional dict of model_id -> outputs
                If None or if any row is missing, will run inference

        Returns:
            TrainingResult with training metrics
        """
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("TRAINING ADAPTIVE ROUTER PROFILE")
        logger.info("=" * 60)
        logger.info(f"Samples: {len(inputs)}")
        logger.info(f"Models: {len(self.models)}")
        logger.info(f"Clusters: {self.n_clusters}")

        # Phase 1: Clustering
        logger.info("\n[1/4] Running clustering...")
        cluster_engine = self._run_clustering(inputs)
        logger.info(
            f"✓ Clustering complete. Silhouette score: {cluster_engine.silhouette:.3f}"
        )

        # Get cluster assignments for error computation
        cluster_assignments = cluster_engine.predict(inputs)

        # Phase 2: Inference (if needed)
        actual_outputs, inference_time = self._ensure_actual_outputs(
            inputs, actual_outputs
        )

        # Phase 3: Compute error rates
        logger.info("\n[3/4] Computing error rates per cluster...")
        error_rates = self._compute_error_rates(
            cluster_engine,
            inputs,
            expected_outputs,
            actual_outputs,
            cluster_assignments,
        )
        logger.info("✓ Error rates computed")

        # Phase 4: Build profile
        logger.info("\n[4/4] Building profile...")
        profile = self._build_profile(cluster_engine, error_rates)
        self._trained_profile = profile
        logger.info("✓ Profile built")

        training_time = time.time() - start_time

        # Get samples per cluster using vectorized bincount
        samples_per_cluster = list(
            np.bincount(cluster_assignments, minlength=self.n_clusters)
        )

        result = TrainingResult(
            n_clusters=self.n_clusters,
            silhouette_score=float(cluster_engine.silhouette),
            n_models=len(self.models),
            model_ids=[m.unique_id() for m in self.models],
            error_rates=error_rates,
            total_samples=len(inputs),
            samples_per_cluster=samples_per_cluster,
            inference_time=inference_time,
            training_time=training_time,
        )

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {training_time:.1f}s")
        logger.info("Profile ready for saving")

        return result

    def _run_clustering(self, inputs: list[str]) -> ClusterEngine:
        """Run clustering on input strings.

        Args:
            inputs: List of input strings to cluster

        Returns:
            Fitted ClusterEngine
        """
        logger.info(f"Fitting {self.n_clusters} clusters on {len(inputs)} inputs")
        self.cluster_engine.fit(inputs)
        return self.cluster_engine

    async def _ensure_actual_outputs_async(
        self, inputs: list[str], actual_outputs: dict[str, list[str]] | None = None
    ) -> tuple[dict[str, list[str]], float]:
        """Async version: Ensure all rows have actual outputs, running inference where needed.

        Args:
            inputs: List of input strings
            actual_outputs: Optional dict of model_id -> outputs (may have missing values)

        Returns:
            Tuple of (complete dict of model_id -> outputs with all values filled, elapsed_seconds)
        """
        # If no actual outputs provided, run full inference
        if actual_outputs is None:
            logger.info(
                "\n[2/4] No actual outputs provided - running full inference..."
            )
            inference_start = time.time()
            result = await self._run_parallel_inference(inputs)
            inference_time = time.time() - inference_start
            logger.info(f"✓ Full inference complete ({inference_time:.1f}s)")
            return result, inference_time

        # Check which models/rows need inference
        n_samples = len(inputs)
        models_needing_inference = []
        indices_needing_inference = {}  # model_id -> list of indices

        for model in self.models:
            model_id = model.unique_id()

            # Check if model has outputs
            if model_id not in actual_outputs:
                models_needing_inference.append(model)
                indices_needing_inference[model_id] = list(range(n_samples))
                continue

            # Check if all rows have outputs
            outputs = actual_outputs[model_id]
            if len(outputs) != n_samples:
                # Length mismatch - need full inference for this model
                models_needing_inference.append(model)
                indices_needing_inference[model_id] = list(range(n_samples))
                continue

            # Check for missing/empty values
            missing_indices = [
                i for i, out in enumerate(outputs) if not out or not out.strip()
            ]

            if missing_indices:
                models_needing_inference.append(model)
                indices_needing_inference[model_id] = missing_indices

        # If no inference needed, return as-is
        if not models_needing_inference:
            logger.info("\n[2/4] All actual outputs present (skipping inference)")
            return actual_outputs, 0.0

        # Run inference for missing values
        logger.info(
            f"\n[2/4] Running partial inference for {len(models_needing_inference)} models..."
        )
        inference_start = time.time()

        # Copy existing outputs
        result = {
            model_id: list(outputs) for model_id, outputs in actual_outputs.items()
        }

        # Create shared semaphore for all models
        semaphore = asyncio.Semaphore(self.max_parallel)

        # Run inference for each model that needs it
        for model in models_needing_inference:
            model_id = model.unique_id()
            indices = indices_needing_inference[model_id]

            logger.info(f"  {model_id}: Inferencing {len(indices)}/{n_samples} samples")

            # Get inputs for missing indices
            missing_inputs = [inputs[i] for i in indices]

            # Run inference
            provider_config = self.provider_configs[model.provider]
            client = self._create_client(model, provider_config)
            outputs = await self._async_generate(client, missing_inputs, semaphore)

            # Ensure result storage matches required length
            if model_id not in result or len(result.get(model_id, [])) != n_samples:
                result[model_id] = [""] * n_samples

            # Fill in the missing outputs
            for idx, output in zip(indices, outputs):
                result[model_id][idx] = output

        inference_time = time.time() - inference_start
        logger.info(f"✓ Partial inference complete ({inference_time:.1f}s)")

        return result, inference_time

    def _ensure_actual_outputs(
        self, inputs: list[str], actual_outputs: dict[str, list[str]] | None = None
    ) -> tuple[dict[str, list[str]], float]:
        """Sync wrapper: Ensure all rows have actual outputs, running inference where needed.

        Args:
            inputs: List of input strings
            actual_outputs: Optional dict of model_id -> outputs (may have missing values)

        Returns:
            Tuple of (complete dict of model_id -> outputs with all values filled, elapsed_seconds)
        """
        return asyncio.run(self._ensure_actual_outputs_async(inputs, actual_outputs))

    async def _run_parallel_inference(self, inputs: list[str]) -> dict[str, list[str]]:
        """Run parallel inference across all models.

        Args:
            inputs: List of input strings

        Returns:
            Dict mapping model_id to list of outputs
        """
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_model(model: Model) -> tuple[str, list[str]]:
            """Run inference for a single model."""
            model_id = model.unique_id()
            logger.info(f"  Running inference with {model_id}...")
            provider_config = self.provider_configs[model.provider]
            client = self._create_client(model, provider_config)

            outputs = await self._async_generate(client, inputs, semaphore)
            return model_id, outputs

        # Run all models concurrently
        model_results = await asyncio.gather(
            *[run_model(model) for model in self.models]
        )

        # Convert to dict
        results = {model_id: outputs for model_id, outputs in model_results}
        return results

    def _create_client(self, model: Model, config: ProviderConfig) -> DeepEvalBaseLLM:
        """Create DeepEval client for model.

        Args:
            model: Model to create client for
            config: Provider configuration

        Returns:
            DeepEval LLM client

        Raises:
            ValueError: If provider not supported or API key missing (except for local)
        """
        factory = default_registry.get_factory(model.provider)

        # Build client kwargs
        kwargs = {"model": model.model_name}

        # Allow empty API key for local provider
        if not config.api_key and model.provider != "local":
            raise ValueError(f"API key is required for provider '{model.provider}'")

        # Add config parameters
        kwargs["api_key"] = config.api_key

        if config.base_url:
            kwargs["base_url"] = config.base_url

        if config.organization:
            kwargs["organization"] = config.organization

        return factory(**kwargs)

    async def _async_generate(
        self,
        client: DeepEvalBaseLLM,
        inputs: list[str],
        semaphore: asyncio.Semaphore,
    ) -> list[str]:
        """Generate outputs asynchronously with rate limiting.

        Rate limiting is applied per individual API call, not per batch.
        This allows concurrent requests across multiple models while
        respecting the global max_parallel limit.

        Args:
            client: DeepEval LLM client
            inputs: List of input prompts
            semaphore: Semaphore for rate limiting individual API calls

        Returns:
            List of output strings
        """
        try:

            async def generate_one(prompt: str) -> str:
                """Generate output for a single prompt with rate limiting."""
                async with semaphore:
                    response = await asyncio.to_thread(client.generate, prompt)
                    return response

            tasks = [generate_one(inp) for inp in inputs]
            return await asyncio.gather(*tasks)
        except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.warning(f"Generation failed with recoverable error: {e}")
            return [""] * len(inputs)
        except Exception as e:
            logger.error(f"Generation failed with unexpected error: {e}")
            raise

    def _compute_error_rates(
        self,
        cluster_engine: ClusterEngine,
        inputs: list[str],
        expected_outputs: list[str],
        actual_outputs: dict[str, list[str]],
        cluster_assignments: np.ndarray,
    ) -> dict[str, list[float]]:
        """Compute error rates per model per cluster.

        Args:
            cluster_engine: Fitted cluster engine
            inputs: Input strings
            expected_outputs: Expected output strings
            actual_outputs: Dict of model_id -> actual outputs

        Returns:
            Dict of model_id -> list of error rates per cluster
        """
        # cluster_assignments passed as parameter

        error_rates = {}

        for model_id, actuals in actual_outputs.items():
            rates = []

            # Use pure Python for string operations (strings don't vectorize well in NumPy)
            expected_clean = [s.strip().lower() for s in expected_outputs]
            actual_clean = [s.strip().lower() for s in actuals]
            correctness = np.array(
                [e == a for e, a in zip(expected_clean, actual_clean)]
            )

            for cluster_id in range(self.n_clusters):
                # Get samples in this cluster
                mask = cluster_assignments == cluster_id

                if not np.any(mask):
                    # Empty cluster - use 0.5 as default error rate
                    rates.append(0.5)
                    continue

                # Compute error rate using vectorized operations
                cluster_correct_fraction = np.mean(correctness[mask])
                error_rate = 1.0 - cluster_correct_fraction
                rates.append(error_rate)

            error_rates[model_id] = rates

            # Log error rates for this model
            avg_error = np.mean(rates)
            logger.info(f"  {model_id}: {avg_error:.2%} avg error rate")

        return error_rates

    def _build_profile(
        self, cluster_engine: ClusterEngine, error_rates: dict[str, list[float]]
    ) -> RouterProfile:
        """Build RouterProfile from clustering and error rates.

        Args:
            cluster_engine: Fitted cluster engine
            error_rates: Error rates per model per cluster

        Returns:
            RouterProfile object
        """
        # Assert components are configured for mypy
        assert cluster_engine.kmeans is not None
        assert cluster_engine.embedding_model is not None

        # Extract cluster centers
        cluster_centers = ClusterCentersData(
            n_clusters=self.n_clusters,
            feature_dim=cluster_engine.kmeans.cluster_centers_.shape[1],
            cluster_centers=cluster_engine.kmeans.cluster_centers_.tolist(),
        )

        # Build metadata with all configurations
        metadata = ProfileMetadata(
            n_clusters=self.n_clusters,
            embedding_model=self.embedding_model,
            dtype=cluster_engine.get_dtype_string(),
            silhouette_score=float(cluster_engine.silhouette),
            clustering=ClusteringConfig(
                max_iter=self.clustering_config.max_iter,
                random_state=self.clustering_config.random_state,
                n_init=self.clustering_config.n_init,
                algorithm=self.clustering_config.algorithm,
                normalization_strategy=self.clustering_config.normalization_strategy,
                n_iter=cluster_engine.kmeans.n_iter_,
            ),
        )

        # Merge error_rates into models
        models_with_errors = []
        for model in self.models:
            model_id = model.unique_id()
            model_with_errors = model.model_copy(
                update={"error_rates": error_rates.get(model_id, [])}
            )
            models_with_errors.append(model_with_errors)

        return RouterProfile(
            cluster_centers=cluster_centers,
            models=models_with_errors,
            metadata=metadata,
        )

    def save_profile(self, path: str | Path) -> Path:
        """Save trained profile to specified path.

        Args:
            path: Destination path (JSON or MessagePack file)

        Returns:
            Path where profile was saved

        Raises:
            ValueError: If no profile has been trained yet
        """
        if self._trained_profile is None:
            raise ValueError(
                "No profile has been trained yet. Call a training method first."
            )

        path = Path(path)

        # Convert pydantic profile to JSON string, then to C++ profile
        from adaptive_core_ext import RouterProfile as CoreProfile

        profile_json = self._trained_profile.model_dump_json()
        core_profile = CoreProfile.from_json_string(profile_json)

        # Use C++ serialization
        if path.suffix.lower() == ".msgpack":
            core_profile.to_msgpack_file(str(path))
        else:
            # Default to JSON for .json or any other extension
            core_profile.to_json_file(str(path))

        logger.info(f"Profile saved to: {path}")
        return path

    @property
    def profile(self) -> RouterProfile:
        """Get the trained profile.

        Returns:
            Trained RouterProfile

        Raises:
            ValueError: If no profile has been trained yet
        """
        if self._trained_profile is None:
            raise ValueError(
                "No profile has been trained yet. Call a training method first."
            )
        return self._trained_profile
