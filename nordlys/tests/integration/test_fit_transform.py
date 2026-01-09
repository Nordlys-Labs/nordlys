"""Integration tests for Nordlys fit and transform methods."""

import numpy as np
import pandas as pd
import pytest

from nordlys import Nordlys


class TestNordlysFit:
    """Test Nordlys fit method."""

    def test_basic_fit(self, three_models, training_data_100):
        """Test fitting Nordlys on sample data."""
        nordlys = Nordlys(models=three_models, nr_clusters=10)
        result = nordlys.fit(training_data_100)

        assert result is nordlys  # Chaining
        assert nordlys._is_fitted is True

    def test_fit_sets_is_fitted_flag(self, three_models, training_data_100):
        """Test that fit sets _is_fitted flag."""
        nordlys = Nordlys(models=three_models)
        assert nordlys._is_fitted is False

        nordlys.fit(training_data_100)
        assert nordlys._is_fitted is True

    def test_fit_initializes_core_engine(self, three_models, training_data_100):
        """Test that fit initializes C++ core engine."""
        nordlys = Nordlys(models=three_models)
        assert nordlys._core_engine is None

        nordlys.fit(training_data_100)
        assert nordlys._core_engine is not None

    def test_fit_computes_embeddings(self, three_models, training_data_100):
        """Test that fit computes embeddings."""
        nordlys = Nordlys(models=three_models)
        nordlys.fit(training_data_100)

        assert nordlys._embeddings is not None
        assert nordlys._embeddings.shape[0] == 100  # n_samples
        assert nordlys._embeddings.shape[1] > 0  # embedding dim

    def test_fit_computes_clusters(self, three_models, training_data_100):
        """Test that fit computes cluster labels."""
        nordlys = Nordlys(models=three_models, nr_clusters=10)
        nordlys.fit(training_data_100)

        assert nordlys._labels is not None
        assert nordlys._labels.shape == (100,)
        assert len(np.unique(nordlys._labels)) <= 10

    def test_fit_computes_centroids(self, three_models, training_data_100):
        """Test that fit computes cluster centroids."""
        nordlys = Nordlys(models=three_models, nr_clusters=10)
        nordlys.fit(training_data_100)

        assert nordlys._centroids is not None
        # Centroids shape depends on whether reducer is used
        assert nordlys._centroids.shape[0] <= 10  # n_clusters

    def test_fit_computes_model_accuracies(self, three_models, training_data_100):
        """Test that fit computes per-cluster model accuracies."""
        nordlys = Nordlys(models=three_models, nr_clusters=10)
        nordlys.fit(training_data_100)

        assert nordlys._model_accuracies is not None
        assert isinstance(nordlys._model_accuracies, dict)

        # Check structure: {cluster_id: {model_id: accuracy}}
        for cluster_id, model_accs in nordlys._model_accuracies.items():
            assert isinstance(cluster_id, int)
            assert isinstance(model_accs, dict)
            for model_id, acc in model_accs.items():
                assert isinstance(model_id, str)
                assert 0 <= acc <= 1

    def test_fit_computes_metrics(self, three_models, training_data_100):
        """Test that fit computes clustering metrics."""
        nordlys = Nordlys(models=three_models)
        nordlys.fit(training_data_100)

        assert nordlys._metrics is not None
        assert nordlys._metrics.n_clusters > 0
        assert nordlys._metrics.n_samples == 100

    def test_fit_returns_self(self, three_models, training_data_100):
        """Test that fit returns self for method chaining."""
        nordlys = Nordlys(models=three_models)
        result = nordlys.fit(training_data_100)
        assert result is nordlys


class TestNordlysFitTransform:
    """Test Nordlys fit_transform method."""

    def test_fit_transform_returns_tuple(self, three_models, training_data_100):
        """Test that fit_transform returns (embeddings, labels)."""
        nordlys = Nordlys(models=three_models)
        result = nordlys.fit_transform(training_data_100)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fit_transform_embeddings_shape(self, three_models, training_data_100):
        """Test that embeddings have correct shape."""
        nordlys = Nordlys(models=three_models)
        embeddings, labels = nordlys.fit_transform(training_data_100)

        assert embeddings.shape[0] == 100
        assert embeddings.shape[1] > 0

    def test_fit_transform_labels_shape(self, three_models, training_data_100):
        """Test that labels have correct shape."""
        nordlys = Nordlys(models=three_models, nr_clusters=10)
        embeddings, labels = nordlys.fit_transform(training_data_100)

        assert labels.shape == (100,)
        assert len(np.unique(labels)) <= 10

    def test_fit_transform_sets_fitted_state(self, three_models, training_data_100):
        """Test that fit_transform sets fitted state."""
        nordlys = Nordlys(models=three_models)
        assert nordlys._is_fitted is False

        nordlys.fit_transform(training_data_100)
        assert nordlys._is_fitted is True


class TestNordlysTransform:
    """Test Nordlys transform method."""

    def test_transform_new_prompts(self, fitted_nordlys):
        """Test transforming new prompts after fitting."""
        new_prompts = [
            "What is machine learning?",
            "Write a sorting algorithm",
            "Explain neural networks",
        ]

        embeddings, labels = fitted_nordlys.transform(new_prompts)

        assert embeddings.shape[0] == 3
        assert labels.shape == (3,)

    def test_transform_returns_embeddings_and_labels(self, fitted_nordlys):
        """Test that transform returns tuple."""
        prompts = ["Test prompt"]
        result = fitted_nordlys.transform(prompts)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_transform_shape_correct(self, fitted_nordlys):
        """Test that transform returns correct shapes."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5"]
        embeddings, labels = fitted_nordlys.transform(prompts)

        assert embeddings.shape[0] == 5
        assert labels.shape == (5,)

    def test_transform_before_fit_raises(self, three_models):
        """Test that transform before fit raises RuntimeError."""
        nordlys = Nordlys(models=three_models)

        with pytest.raises(RuntimeError, match="must be fitted before use"):
            nordlys.transform(["Test prompt"])


class TestNordlysFitValidation:
    """Test Nordlys fit validation."""

    def test_fit_missing_questions_column_raises(self, three_models):
        """Test that fit fails when questions column is missing."""
        nordlys = Nordlys(models=three_models)
        df = pd.DataFrame(
            {
                "prompts": ["Test"],  # Wrong column name
                "openai/gpt-4": [0.9],
                "openai/gpt-3.5-turbo": [0.8],
                "anthropic/claude-3-sonnet": [0.85],
            }
        )

        with pytest.raises(ValueError, match="must have a 'questions' column"):
            nordlys.fit(df)

    def test_fit_missing_model_columns_raises(self, three_models):
        """Test that fit fails when model columns are missing."""
        nordlys = Nordlys(models=three_models)
        df = pd.DataFrame(
            {
                "questions": ["Test"],
                "openai/gpt-4": [0.9],
                # Missing gpt-3.5-turbo and claude-3-sonnet
            }
        )

        with pytest.raises(ValueError, match="missing accuracy columns"):
            nordlys.fit(df)

    def test_fit_empty_dataframe_raises(self, three_models):
        """Test that fit on empty DataFrame raises error."""
        nordlys = Nordlys(models=three_models)
        df = pd.DataFrame(
            {
                "questions": [],
                "openai/gpt-4": [],
                "openai/gpt-3.5-turbo": [],
                "anthropic/claude-3-sonnet": [],
            }
        )

        # Empty DataFrame causes ValueError (empty array in embedding/clustering)
        with pytest.raises(ValueError, match="array"):
            nordlys.fit(df)

    def test_fit_with_wrong_column_names_raises(self, three_models):
        """Test that fit validates model column names."""
        nordlys = Nordlys(models=three_models)
        df = pd.DataFrame(
            {
                "questions": ["Test"],
                "wrong/model": [0.9],  # Wrong model ID
                "openai/gpt-3.5-turbo": [0.8],
                "anthropic/claude-3-sonnet": [0.85],
            }
        )

        with pytest.raises(ValueError, match="missing accuracy columns"):
            nordlys.fit(df)


class TestNordlysFittedAttributes:
    """Test Nordlys fitted attribute accessors."""

    def test_centroids_property(self, fitted_nordlys):
        """Test centroids_ property."""
        centroids = fitted_nordlys.centroids_
        assert centroids is not None
        assert centroids.ndim == 2
        assert centroids.shape[0] > 0  # n_clusters

    def test_labels_property(self, fitted_nordlys):
        """Test labels_ property."""
        labels = fitted_nordlys.labels_
        assert labels is not None
        assert labels.ndim == 1
        assert len(labels) == 100  # n_samples from training

    def test_embeddings_property(self, fitted_nordlys):
        """Test embeddings_ property."""
        embeddings = fitted_nordlys.embeddings_
        assert embeddings is not None
        assert embeddings.shape[0] == 100  # n_samples

    def test_metrics_property(self, fitted_nordlys):
        """Test metrics_ property."""
        metrics = fitted_nordlys.metrics_
        assert metrics is not None
        assert metrics.n_clusters > 0
        assert metrics.n_samples == 100

    def test_model_accuracies_property(self, fitted_nordlys):
        """Test model_accuracies_ property."""
        accs = fitted_nordlys.model_accuracies_
        assert accs is not None
        assert isinstance(accs, dict)

    def test_n_clusters_property(self, fitted_nordlys):
        """Test n_clusters_ property."""
        n = fitted_nordlys.n_clusters_
        assert isinstance(n, int)
        assert n == 10  # As specified in fixture

    def test_properties_before_fit_raise(self, three_models):
        """Test that properties raise error before fit."""
        nordlys = Nordlys(models=three_models)

        with pytest.raises(RuntimeError):
            _ = nordlys.centroids_

        with pytest.raises(RuntimeError):
            _ = nordlys.labels_

        with pytest.raises(RuntimeError):
            _ = nordlys.embeddings_

        with pytest.raises(RuntimeError):
            _ = nordlys.metrics_

        with pytest.raises(RuntimeError):
            _ = nordlys.model_accuracies_

        with pytest.raises(RuntimeError):
            _ = nordlys.n_clusters_
