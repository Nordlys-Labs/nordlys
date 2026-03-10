"""Integration tests for Dataset-native Trainer."""

from __future__ import annotations

import numpy as np
import pytest

from nordlys import Dataset, ModelConfig, Nordlys, Trainer


@pytest.fixture
def trainer_models() -> list[ModelConfig]:
    return [
        ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
        ModelConfig(id="openai/gpt-3.5-turbo", cost_input=0.5, cost_output=1.5),
        ModelConfig(id="anthropic/claude-3-sonnet", cost_input=15.0, cost_output=75.0),
    ]


@pytest.fixture
def trainer_dataset(trainer_models: list[ModelConfig]) -> Dataset:
    rng = np.random.default_rng(42)
    rows = []
    model_ids = [model.id for model in trainer_models]
    for idx in range(45):
        targets = {
            model_ids[0]: int(rng.random() > 0.30),
            model_ids[1]: int(rng.random() > 0.55),
            model_ids[2]: int(rng.random() > 0.40),
        }
        rows.append(
            {
                "id": f"row-{idx}",
                "input": f"Prompt number {idx}: explain concept {idx % 7}",
                "targets": targets,
            }
        )
    return Dataset.from_list(rows)


class TestTrainer:
    def test_fit_returns_checkpoint(
        self, trainer_models: list[ModelConfig], trainer_dataset: Dataset
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer="kmeans", n_clusters=8)
        checkpoint = trainer.fit(trainer_dataset)
        assert checkpoint is not None
        assert checkpoint.version == "2.0"

    def test_checkpoint_loadable_by_router(
        self, trainer_models: list[ModelConfig], trainer_dataset: Dataset
    ) -> None:
        trainer = Trainer(models=trainer_models, clusterer="kmeans", n_clusters=8)
        checkpoint = trainer.fit(trainer_dataset)

        router = Nordlys._from_checkpoint(
            checkpoint, models=trainer_models, device="cpu"
        )
        result = router.route("Explain backtracking with example")
        assert result.model_id in {model.id for model in trainer_models}

    def test_missing_targets_column_raises(
        self, trainer_models: list[ModelConfig]
    ) -> None:
        dataset = Dataset.from_list([{"id": "1", "input": "hello"}])
        trainer = Trainer(models=trainer_models)
        with pytest.raises(ValueError, match="Dataset validation failed"):
            trainer.fit(dataset)
