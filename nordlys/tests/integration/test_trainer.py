"""Integration tests for Dataset-native Trainer."""

from __future__ import annotations

import numpy as np
import pytest

from nordlys import Dataset, Trainer
from nordlys.clustering import HDBSCANClusterer, KMeansClusterer


@pytest.fixture(autouse=True)
def mock_trainer_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEmbedder:
        def encode(self, texts: list[str]) -> np.ndarray:
            vectors = []
            for text in texts:
                seed = sum(ord(ch) for ch in text) % (2**32)
                rng = np.random.default_rng(seed)
                group = seed % 8
                center = np.zeros(384, dtype=np.float32)
                center[group] = 10.0
                noise = rng.normal(0.0, 0.05, size=384).astype(np.float32)
                vectors.append(center + noise)
            return np.asarray(vectors, dtype=np.float32)

        def checkpoint_config(self) -> dict[str, str | bool]:
            return {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "trust_remote_code": False,
            }

    monkeypatch.setattr(Trainer, "_make_embedder", lambda self: FakeEmbedder())


@pytest.fixture(autouse=True)
def mock_router_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSentenceTransformer:
        def __init__(
            self,
            model: str,
            device: str = "cpu",
            trust_remote_code: bool = False,
        ) -> None:
            self.model = model
            self.device = device
            self.trust_remote_code = trust_remote_code
            self.max_seq_length = 0
            self.tokenizer = type(
                "Tokenizer", (), {"clean_up_tokenization_spaces": True}
            )()

        def encode(
            self,
            texts: list[str],
            convert_to_numpy: bool = True,
            show_progress_bar: bool = False,
            normalize_embeddings: bool = True,
            prompt_name: str | None = None,
            prompt: str | None = None,
        ) -> np.ndarray:
            vectors = []
            for text in texts:
                seed = sum(ord(ch) for ch in text) % (2**32)
                rng = np.random.default_rng(seed)
                group = seed % 8
                center = np.zeros(384, dtype=np.float32)
                center[group] = 10.0
                noise = rng.normal(0.0, 0.05, size=384).astype(np.float32)
                vectors.append(center + noise)
            return np.asarray(vectors, dtype=np.float32)

    monkeypatch.setattr("nordlys.router.SentenceTransformer", FakeSentenceTransformer)


@pytest.fixture
def trainer_models() -> list[str]:
    return [
        "gpt-4",
        "gpt-3.5-turbo",
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


@pytest.fixture
def test_hdbscan_clusterer() -> HDBSCANClusterer:
    return HDBSCANClusterer(min_cluster_size=2, min_samples=1)


class TestTrainerInitialization:
    def test_default_initialization(self, trainer_models: list[str]) -> None:
        trainer = Trainer(models=trainer_models)
        assert trainer.models == trainer_models
        assert trainer.input_col == "input"
        assert trainer.target_col == "targets"
        assert trainer.clusterer is None

    def test_custom_columns(self, trainer_models: list[str]) -> None:
        trainer = Trainer(
            models=trainer_models,
            input_col="prompt",
            target_col="labels",
        )
        assert trainer.input_col == "prompt"
        assert trainer.target_col == "labels"


class TestTrainerFit:
    def test_fit_with_hdbscan_medium(
        self,
        trainer_models: list[str],
        medium_dataset: Dataset,
        test_hdbscan_clusterer: HDBSCANClusterer,
    ) -> None:
        """HDBSCAN on medium dataset."""
        trainer = Trainer(models=trainer_models, clusterer=test_hdbscan_clusterer)
        fitted = trainer.fit_structure(medium_dataset)
        assert fitted is not None
        assert fitted.n_clusters > 0

    def test_fit_with_kmeans_large(
        self, trainer_models: list[str], large_dataset: Dataset
    ) -> None:
        """KMeans on large dataset with many clusters."""
        trainer = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=10, random_state=42),
        )
        fitted = trainer.fit_structure(large_dataset)
        assert fitted.n_clusters == 10

    def test_fit_with_custom_clusterer(
        self, trainer_models: list[str], large_dataset: Dataset
    ) -> None:
        """Custom clusterer instance."""
        trainer = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=5, random_state=42),
        )
        fitted = trainer.fit_structure(large_dataset)
        assert fitted.n_clusters == 5


class TestTrainerValidation:
    def test_empty_models_raises(self) -> None:
        """Test that empty models list raises."""
        trainer = Trainer(models=[])
        dataset = Dataset.from_list([{"id": "1", "input": "test", "targets": {"a": 1}}])
        with pytest.raises(ValueError, match="At least one model"):
            trainer.fit_structure(dataset)

    def test_missing_input_column_raises(self, trainer_models: list[str]) -> None:
        """Test that missing input column raises."""
        dataset = Dataset.from_list(
            [{"id": "1", "text": "hello", "targets": {"gpt-4": 1}}]
        )
        trainer = Trainer(models=trainer_models)
        with pytest.raises(ValueError, match="Dataset validation failed"):
            trainer.fit_structure(dataset)


class TestTrainerCheckpoint:
    def test_checkpoint_has_cluster_centers(
        self,
        trainer_models: list[str],
        medium_dataset: Dataset,
        test_hdbscan_clusterer: HDBSCANClusterer,
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer=test_hdbscan_clusterer)
        fitted = trainer.fit_structure(medium_dataset)
        assert fitted.n_clusters > 0

    def test_checkpoint_has_models(
        self,
        trainer_models: list[str],
        medium_dataset: Dataset,
        test_hdbscan_clusterer: HDBSCANClusterer,
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer=test_hdbscan_clusterer)
        fitted = trainer.fit_structure(medium_dataset)
        scored = trainer.calibrate(fitted, medium_dataset)
        assert len(scored.scores) == len(trainer_models)

    def test_checkpoint_scores_shape(
        self,
        trainer_models: list[str],
        medium_dataset: Dataset,
        test_hdbscan_clusterer: HDBSCANClusterer,
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer=test_hdbscan_clusterer)
        fitted = trainer.fit_structure(medium_dataset)
        scored = trainer.calibrate(fitted, medium_dataset)
        n_clusters = fitted.n_clusters
        for model_scores in scored.scores.values():
            assert len(model_scores) == n_clusters


class TestTrainerRouterIntegration:
    def test_checkpoint_loadable_by_router(
        self,
        trainer_models: list[str],
        medium_dataset: Dataset,
        test_hdbscan_clusterer: HDBSCANClusterer,
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer=test_hdbscan_clusterer)
        fitted = trainer.fit_structure(medium_dataset)
        router = trainer.compile(fitted, trainer.calibrate(fitted, medium_dataset))
        result = router.route("Explain backtracking with example")
        assert result.model_id in set(trainer_models)

    def test_router_route_batch(
        self, trainer_models: list[str], large_dataset: Dataset
    ) -> None:
        trainer = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=10, random_state=42),
        )
        fitted = trainer.fit_structure(large_dataset)
        router = trainer.compile(fitted, trainer.calibrate(fitted, large_dataset))
        prompts = ["Explain quantum", "Write code", "What is ML?"]
        results = router.route_batch(prompts)
        assert len(results) == 3
        for r in results:
            assert r.model_id in set(trainer_models)


class TestTrainerHyperparameters:
    def test_hdbscan_params(
        self, trainer_models: list[str], medium_dataset: Dataset
    ) -> None:
        trainer = Trainer(
            models=trainer_models,
            clusterer=HDBSCANClusterer(min_cluster_size=10, min_samples=3),
        )
        fitted = trainer.fit_structure(medium_dataset)
        assert fitted is not None

    def test_kmeans_params(
        self, trainer_models: list[str], large_dataset: Dataset
    ) -> None:
        trainer = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(
                n_clusters=15,
                max_iter=200,
                n_init=5,
                random_state=42,
            ),
        )
        fitted = trainer.fit_structure(large_dataset)
        assert fitted.n_clusters == 15

    def test_reducer_serialized(
        self, trainer_models: list[str], large_dataset: Dataset
    ) -> None:
        """Test that reducer-backed checkpoints persist reduction metadata."""
        from nordlys.reduction import PCAReducer

        trainer = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=8, random_state=42),
            reducer=PCAReducer(n_components=8, random_state=42),
        )
        fitted = trainer.fit_structure(large_dataset)

        assert fitted.reduction is not None
        assert fitted.reduction.kind == "pca"

    def test_router_restores_reducer(
        self, trainer_models: list[str], large_dataset: Dataset
    ) -> None:
        """Test that router restores a serialized reducer before routing."""
        from nordlys.reduction import PCAReducer

        trainer = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=8, random_state=42),
            reducer=PCAReducer(n_components=8, random_state=42),
        )
        fitted = trainer.fit_structure(large_dataset)
        router = trainer.compile(fitted, trainer.calibrate(fitted, large_dataset))
        result = router.route("Explain backtracking with example")

        assert fitted.reduction is not None
        assert result.model_id in set(trainer_models)


class TestTrainerDeterminism:
    def test_same_seed_same_result(
        self, trainer_models: list[str], large_dataset: Dataset
    ) -> None:
        trainer1 = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=10, random_state=42),
            random_state=42,
        )
        trainer2 = Trainer(
            models=trainer_models,
            clusterer=KMeansClusterer(n_clusters=10, random_state=42),
            random_state=42,
        )
        fitted1 = trainer1.fit_structure(large_dataset)
        fitted2 = trainer2.fit_structure(large_dataset)
        np.testing.assert_array_almost_equal(
            fitted1.cluster_centers, fitted2.cluster_centers
        )
