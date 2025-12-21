#!/usr/bin/env python3
"""Adaptive Router Training Script with TOML Configuration.

This script trains an Adaptive Router profile from labeled dataset files using
TOML-based configuration. It supports hybrid model loading (fetch pricing from
API or define in TOML) and multiple output formats (local, S3, MinIO).

Usage:
    python train.py --config config.toml

Example TOML configurations can be found in the train/examples/ directory.
"""

import sys
import logging
import argparse
from typing import Dict, List

import httpx
import polars as pl

# Import from adaptive_router library
from adaptive_router.core.trainer import Trainer
from adaptive_router.models.api import Model
from adaptive_router.models.train import ProviderConfig


# Import from local train module
from train.config import TrainingConfig, load_config, ModelConfigToml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AdaptiveModelsAPIClient:
    """HTTP client for the Adaptive Models API.

    Handles authentication and communication with the Adaptive Models API
    to fetch model information and pricing data.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.llmadaptive.uk/v1",
        timeout: float = 30.0,
    ):
        """Initialize the API client.

        Args:
            api_key: Adaptive API key for authentication
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def get_model_info(self, provider: str, model_name: str) -> dict:
        """Fetch model information from the API.

        Args:
            provider: Model provider (e.g., "openai", "anthropic")
            model_name: Model name (e.g., "gpt-4", "claude-3-5-sonnet-20241022")

        Returns:
            Model information dictionary from API

        Raises:
            ValueError: If model not found or API errors occur
        """
        url = f"{self.base_url}/models/{provider}/{model_name}"

        try:
            response = self.client.get(url, headers=self.headers)

            if response.status_code == 404:
                raise ValueError(f"Model not found: {provider}/{model_name}")

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            raise ValueError(f"API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Network error: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close HTTP client."""
        self.client.close()


def create_model_from_toml(model_config: ModelConfigToml) -> Model:
    """Create a Model object from TOML configuration (pricing included).

    Args:
        model_config: Model configuration from TOML

    Returns:
        Model object with pricing from TOML
    """
    # Assert pricing is available (should be guaranteed by caller)
    assert model_config.cost_per_1m_input_tokens is not None
    assert model_config.cost_per_1m_output_tokens is not None

    logger.info(
        f"Using TOML pricing for {model_config.model_spec()}: "
        f"${model_config.cost_per_1m_input_tokens:.4f}/1M input tokens, "
        f"${model_config.cost_per_1m_output_tokens:.4f}/1M output tokens"
    )

    return Model(
        provider=model_config.provider,
        model_name=model_config.model_name,
        cost_per_1m_input_tokens=model_config.cost_per_1m_input_tokens,
        cost_per_1m_output_tokens=model_config.cost_per_1m_output_tokens,
    )


def create_model_from_api(
    api_client: AdaptiveModelsAPIClient, model_config: ModelConfigToml
) -> Model:
    """Create a Model object by fetching pricing from the API.

    Args:
        api_client: Initialized API client
        model_config: Model configuration from TOML (without pricing)

    Returns:
        Model object with pricing from API

    Raises:
        ValueError: If model not found or pricing data is invalid
    """
    # Fetch from API
    data = api_client.get_model_info(model_config.provider, model_config.model_name)

    # Extract pricing
    pricing = data.get("pricing")
    if not pricing:
        raise ValueError(f"No pricing information for {model_config.model_spec()}")

    prompt_cost = pricing.get("prompt_cost")
    completion_cost = pricing.get("completion_cost")

    if prompt_cost is None or completion_cost is None:
        raise ValueError(
            f"Incomplete pricing data for {model_config.model_spec()}. "
            f"Both prompt_cost and completion_cost are required."
        )

    # Convert pricing (API returns per-token, Model expects per-1M-tokens)
    cost_per_1m_input = float(prompt_cost) * 1_000_000
    cost_per_1m_output = float(completion_cost) * 1_000_000

    logger.info(
        f"Fetched API pricing for {model_config.model_spec()}: "
        f"${cost_per_1m_input:.4f}/1M input tokens, "
        f"${cost_per_1m_output:.4f}/1M output tokens"
    )

    return Model(
        provider=model_config.provider,
        model_name=model_config.model_name,
        cost_per_1m_input_tokens=cost_per_1m_input,
        cost_per_1m_output_tokens=cost_per_1m_output,
    )


def load_models(
    config: TrainingConfig, api_client: AdaptiveModelsAPIClient
) -> List[Model]:
    """Load models from TOML configuration (hybrid mode).

    For models with pricing in TOML, create directly.
    For models without pricing, fetch from API.

    Args:
        config: Validated training configuration
        api_client: API client for fetching model pricing

    Returns:
        List of Model objects

    Raises:
        ValueError: If any model fails to load
    """
    models = []
    failed = []

    for model_config in config.models:
        try:
            if model_config.has_pricing:
                # Use TOML pricing
                model = create_model_from_toml(model_config)
            else:
                # Fetch from API
                model = create_model_from_api(api_client, model_config)

            models.append(model)

        except Exception as e:
            logger.error(f"Failed to load {model_config.model_spec()}: {e}")
            failed.append((model_config.model_spec(), str(e)))

    if failed:
        logger.error(f"\nFailed to load {len(failed)} model(s):")
        for spec, error in failed:
            logger.error(f"  - {spec}: {error}")
        raise ValueError(f"Failed to load {len(failed)} model(s)")

    return models


def convert_provider_configs(config: TrainingConfig) -> Dict[str, ProviderConfig]:
    """Convert TOML provider configs to ProviderConfig objects.

    Args:
        config: Validated training configuration

    Returns:
        Dictionary mapping provider names to ProviderConfig objects
    """
    provider_configs = {}

    for provider_name, toml_config in config.providers.items():
        provider_configs[provider_name] = ProviderConfig(
            api_key=toml_config.api_key,
            base_url=toml_config.base_url,
            organization=toml_config.organization,
            timeout=toml_config.timeout,
            max_retries=toml_config.max_retries,
        )

    # Add default 'local' provider only when user has not defined one
    if (
        any(m.provider == "local" for m in config.models)
        and "local" not in provider_configs
    ):
        provider_configs["local"] = ProviderConfig(
            api_key="",
            base_url=None,
            organization=None,
            timeout=60.0,
            max_retries=3,
        )

    return provider_configs


def train_router(config: TrainingConfig) -> None:
    """Orchestrate the complete training workflow.

    Args:
        config: Validated training configuration
    """
    # Log training header
    logger.info("=" * 80)
    logger.info("ADAPTIVE ROUTER TRAINING")
    logger.info("=" * 80)
    logger.info("Config file validated successfully")
    logger.info(f"Dataset: {config.dataset.path} ({config.dataset.type})")
    logger.info(f"Models: {len(config.models)}")
    for m in config.models:
        logger.info(f"  - {m.model_spec()}")
    logger.info(f"Output: {config.output.path} ({config.output.storage_type})")
    logger.info(f"Clusters: {config.training.n_clusters}")
    logger.info("=" * 80)

    # Initialize API client
    with AdaptiveModelsAPIClient(
        config.api.adaptive_api_key, config.api.base_url
    ) as api_client:
        # Load models (hybrid mode: TOML or API)
        logger.info("\nLoading models...")
        try:
            models = load_models(config, api_client)
            logger.info(f"Successfully loaded {len(models)} model(s)")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            sys.exit(1)

        # Convert provider configurations
        provider_configs = convert_provider_configs(config)
        logger.info(f"Loaded configurations for {len(provider_configs)} provider(s)")

        # Initialize trainer
        trainer = Trainer(
            models=models,
            provider_configs=provider_configs,
            n_clusters=config.training.n_clusters,
            max_parallel=config.training.max_parallel,
            embedding_model=config.training.embedding_model,
            random_seed=config.training.random_seed,
        )

        # Train based on dataset type
        logger.info(f"\nStarting training from {config.dataset.type} file...")

        try:
            if config.dataset.type == "csv":
                result = trainer.train_from_csv(
                    config.dataset.path,
                    config.dataset.input_column,
                    config.dataset.expected_column,
                )
            elif config.dataset.type == "json":
                result = trainer.train_from_json(
                    config.dataset.path,
                    config.dataset.input_column,
                    config.dataset.expected_column,
                )
            elif config.dataset.type == "parquet":
                df = pl.read_parquet(config.dataset.path)
                result = trainer.train_from_dataframe(
                    df,
                    config.dataset.input_column,
                    config.dataset.expected_column,
                )
            else:
                raise ValueError(f"Unsupported dataset type: {config.dataset.type}")

        except FileNotFoundError:
            logger.error(f"Dataset file not found: {config.dataset.path}")
            sys.exit(1)
        except KeyError as e:
            logger.error(f"Column not found in dataset: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            sys.exit(1)

        # Display training results
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total samples: {result.total_samples}")
        logger.info(f"Number of clusters: {result.n_clusters}")
        logger.info(f"Silhouette score: {result.silhouette_score:.4f}")
        logger.info(f"Training time: {result.training_time:.2f} seconds")
        if result.inference_time:
            logger.info(f"Inference time: {result.inference_time:.2f} seconds")

        # Display error rates
        logger.info("\nModel Error Rates:")
        for model_id, error_rates in result.error_rates.items():
            avg_error = sum(error_rates) / len(error_rates) if error_rates else 0
            logger.info(f"  {model_id}: {avg_error * 100:.2f}%")

        # Save profile
        logger.info("\nSaving profile...")
        try:
            trainer.save_profile(config.output.path)
            logger.info(f"Profile saved to: {config.output.path}")

        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            sys.exit(1)

        # Log completion
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)


def main() -> None:
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description="Train an Adaptive Router profile from TOML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   # Train with minimal configuration
   python train.py --config train/examples/configs/train_minimal.toml

   # Train with custom clustering parameters
   python train.py --config train/examples/configs/train_custom_params.toml

   # Train with S3/MinIO output
   python train.py --config train/examples/configs/train_s3.toml

 See train/examples/ directory for sample TOML configurations.
 """,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to TOML configuration file",
    )

    args = parser.parse_args()

    # Load and validate configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Configuration error: {e}", exc_info=True)
        sys.exit(1)

    # Train router
    train_router(config)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
