"""Integration tests for Dataset-native Trainer."""

from __future__ import annotations

import numpy as np
import pytest

from nordlys import Dataset, ModelConfig, Nordlys, Trainer
from nordlys.clustering import KMeansClusterer


@pytest.fixture
def trainer_models() -> list[ModelConfig]:
    return [
        ModelConfig(id="gpt-4", cost_input=30.0, cost_output=60.0),
        ModelConfig(id="gpt-3.5-turbo", cost_input=0.5, cost_output=1.5),
    ]


@pytest.fixture
def small_dataset() -> Dataset:
    """Small dataset - use HDBSCAN for automatic clustering."""
    return Dataset.from_list(
        [
            {
                "id": "1",
                "input": "Explain quantum computing",
                "targets": {"gpt-4": 1, "gpt-3.5-turbo": 0},
            },
            {
                "id": "2",
                "input": "Write hello world in python",
                "targets": {"gpt-4": 1, "gpt-3.5-turbo": 1},
            },
            {
                "id": "3",
                "input": "What is 2+2?",
                "targets": {"gpt-4": 1, "gpt-3.5-turbo": 1},
            },
            {
                "id": "4",
                "input": "Implement quicksort",
                "targets": {"gpt-4": 1, "gpt-3.5-turbo": 0},
            },
            {
                "id": "5",
                "input": "Explain relativity",
                "targets": {"gpt-4": 1, "gpt-3.5-turbo": 0},
            },
        ]
    )


@pytest.fixture
def medium_dataset() -> Dataset:
    """Medium dataset - use HDBSCAN for automatic clustering."""
    rng = np.random.default_rng(42)
    rows = []
    for idx in range(100):
        rows.append(
            {
                "id": f"row-{idx}",
                "input": f"Prompt number {idx}: explain concept {idx % 10}",
                "targets": {
                    "gpt-4": int(rng.random() > 0.25),
                    "gpt-3.5-turbo": int(rng.random() > 0.45),
                },
            }
        )
    return Dataset.from_list(rows)


@pytest.fixture
def large_dataset() -> Dataset:
    """Large dataset - suitable for kmeans with many clusters."""
    rng = np.random.default_rng(123)
    rows = []
    for idx in range(500):
        rows.append(
            {
                "id": f"row-{idx}",
                "input": f"Task {idx}: {['explain', 'write', 'implement', 'describe'][idx % 4]} {['AI', 'code', 'algorithm', 'system'][idx % 4]}",
                "targets": {
                    "gpt-4": int(rng.random() > 0.20),
                    "gpt-3.5-turbo": int(rng.random() > 0.40),
                },
            }
        )
    return Dataset.from_list(rows)


class TestTrainerInitialization:
    def test_default_initialization(self, trainer_models: list[ModelConfig]) -> None:
        trainer = Trainer(models=trainer_models)
        assert trainer.models == trainer_models
        assert trainer.input_col == "input"
        assert trainer.target_col == "targets"
        assert trainer.clusterer is None

    def test_custom_columns(self, trainer_models: list[ModelConfig]) -> None:
        trainer = Trainer(
            models=trainer_models,
            input_col="prompt",
            target_col="labels",
        )
        assert trainer.input_col == "prompt"
        assert trainer.target_col == "labels"


class TestTrainerFit:
    def test_fit_with_hdbscan_medium(
        self, trainer_models: list[ModelConfig], medium_dataset: Dataset
    ) -> None:
        """HDBSCAN on medium dataset."""
        trainer = Trainer(models=trainer_models, clusterer="hdbscan")
        checkpoint = trainer.fit(medium_dataset)
        assert checkpoint is not None
        assert len(checkpoint.cluster_centers) > 0

    def test_fit_with_kmeans_large(
        self, trainer_models: list[ModelConfig], large_dataset: Dataset
    ) -> None:
        """KMeans on large dataset with many clusters."""
        trainer = Trainer(models=trainer_models, clusterer="kmeans", n_clusters=10)
        checkpoint = trainer.fit(large_dataset)
        assert len(checkpoint.cluster_centers) == 10

    def test_fit_with_custom_clusterer(
        self, trainer_models: list[ModelConfig], large_dataset: Dataset
    ) -> None:
        """Custom clusterer instance."""
        trainer = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=5, random_state=42),
        )
        checkpoint = trainer.fit(large_dataset)
        assert len(checkpoint.cluster_centers) == 5


class TestTrainerValidation:
    def test_empty_models_raises(self) -> None:
        """Test that empty models list raises."""
        trainer = Trainer(models=[])
        dataset = Dataset.from_list([{"id": "1", "input": "test", "targets": {"a": 1}}])
        with pytest.raises(ValueError, match="At least one model"):
            trainer.fit(dataset)

    def test_missing_input_column_raises(
        self, trainer_models: list[ModelConfig]
    ) -> None:
        """Test that missing input column raises."""
        dataset = Dataset.from_list(
            [{"id": "1", "text": "hello", "targets": {"gpt-4": 1}}]
        )
        trainer = Trainer(models=trainer_models)
        with pytest.raises(ValueError, match="Dataset validation failed"):
            trainer.fit(dataset)


class TestTrainerCheckpoint:
    def test_checkpoint_has_cluster_centers(
        self, trainer_models: list[ModelConfig], medium_dataset: Dataset
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer="hdbscan")
        checkpoint = trainer.fit(medium_dataset)
        assert len(checkpoint.cluster_centers) > 0

    def test_checkpoint_has_models(
        self, trainer_models: list[ModelConfig], medium_dataset: Dataset
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer="hdbscan")
        checkpoint = trainer.fit(medium_dataset)
        assert len(checkpoint.models) == len(trainer_models)

    def test_checkpoint_error_rates_shape(
        self, trainer_models: list[ModelConfig], medium_dataset: Dataset
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer="hdbscan")
        checkpoint = trainer.fit(medium_dataset)
        n_clusters = len(checkpoint.cluster_centers)
        for model in checkpoint.models:
            assert len(model.error_rates) == n_clusters


class TestTrainerRouterIntegration:
    def test_checkpoint_loadable_by_router(
        self, trainer_models: list[ModelConfig], medium_dataset: Dataset
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer="hdbscan")
        checkpoint = trainer.fit(medium_dataset)
        router = Nordlys._from_checkpoint(
            checkpoint, models=trainer_models, device="cpu"
        )
        result = router.route("Explain backtracking with example")
        assert result.model_id in {m.id for m in trainer_models}

    def test_router_route_batch(
        self, trainer_models: list[ModelConfig], large_dataset: Dataset
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer="kmeans", n_clusters=10)
        checkpoint = trainer.fit(large_dataset)
        router = Nordlys._from_checkpoint(
            checkpoint, models=trainer_models, device="cpu"
        )
        prompts = ["Explain quantum", "Write code", "What is ML?"]
        results = router.route_batch(prompts)
        assert len(results) == 3
        for r in results:
            assert r.model_id in {m.id for m in trainer_models}


class TestTrainerHyperparameters:
    def test_hdbscan_params(
        self, trainer_models: list[ModelConfig], medium_dataset: Dataset
    ) -> None:
        trainer = Trainer(
            models=trainer_models,
            clusterer="hdbscan",
            hdbscan_min_cluster_size=10,
            hdbscan_min_samples=3,
        )
        checkpoint = trainer.fit(medium_dataset)
        assert checkpoint is not None

    def test_kmeans_params(
        self, trainer_models: list[ModelConfig], large_dataset: Dataset
    ) -> None:
        trainer = Trainer(
            models=trainer_models,
            clusterer="kmeans",
            n_clusters=15,
            kmeans_max_iter=200,
            kmeans_n_init=5,
        )
        checkpoint = trainer.fit(large_dataset)
        assert len(checkpoint.cluster_centers) == 15

    def test_reducer_rejected(
        self, trainer_models: list[ModelConfig], large_dataset: Dataset
    ) -> None:
        """Test that using a reducer raises an error (not supported for checkpoints)."""
        from nordlys.reduction import UMAPReducer

        trainer = Trainer(
            models=trainer_models,
            clusterer="kmeans",
            n_clusters=8,
            reducer=UMAPReducer(n_components=8, random_state=42),
        )
        with pytest.raises(ValueError, match="Reducer is not supported"):
            trainer.fit(large_dataset)


class TestTrainerDeterminism:
    def test_same_seed_same_result(
        self, trainer_models: list[ModelConfig], large_dataset: Dataset
    ) -> None:
        trainer1 = Trainer(
            models=trainer_models,
            clusterer="kmeans",
            n_clusters=10,
            random_state=42,
        )
        trainer2 = Trainer(
            models=trainer_models,
            clusterer="kmeans",
            n_clusters=10,
            random_state=42,
        )
        cp1 = trainer1.fit(large_dataset)
        cp2 = trainer2.fit(large_dataset)
        np.testing.assert_array_almost_equal(cp1.cluster_centers, cp2.cluster_centers)
