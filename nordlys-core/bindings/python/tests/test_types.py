"""Tests for Python bindings - Type classes (ModelFeatures, TrainingMetrics, etc.)."""

import pytest


class TestModelFeatures:
    """Test ModelFeatures type bindings."""

    def test_model_features_from_checkpoint(self, sample_checkpoint_json: str):
        """Test accessing ModelFeatures from checkpoint."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        model = checkpoint.models[0]

        # Test basic properties
        assert isinstance(model.model_id, str)
        assert isinstance(model.cost_per_1m_input_tokens, float)
        assert model.cost_per_1m_input_tokens > 0
        assert isinstance(model.cost_per_1m_output_tokens, float)
        assert model.cost_per_1m_output_tokens > 0

        # Test error_rates vector - this will fail if vector.h is missing in types.cpp
        error_rates = model.error_rates
        assert isinstance(error_rates, list) or hasattr(error_rates, "__iter__")
        error_rates_list = list(error_rates)
        assert len(error_rates_list) == checkpoint.n_clusters
        assert all(isinstance(rate, float) for rate in error_rates_list)
        assert all(0.0 <= rate <= 1.0 for rate in error_rates_list)

    def test_error_rates_vector_operations(self, sample_checkpoint_json: str):
        """Test that error_rates vector supports list operations."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        model = checkpoint.models[0]

        # Test list conversion
        error_rates_list = list(model.error_rates)
        assert isinstance(error_rates_list, list)

        # Test iteration
        for rate in model.error_rates:
            assert isinstance(rate, float)
            assert 0.0 <= rate <= 1.0

        # Test indexing
        assert isinstance(model.error_rates[0], float)
        assert len(model.error_rates) == checkpoint.n_clusters

        # Test membership (if applicable)
        if model.error_rates:
            first_rate = model.error_rates[0]
            assert first_rate in model.error_rates

    def test_model_features_methods(self, sample_checkpoint_json: str):
        """Test ModelFeatures utility methods."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        model = checkpoint.models[0]

        # Test provider extraction
        provider = model.provider()
        assert isinstance(provider, str)
        assert "/" in model.model_id
        assert model.model_id.startswith(provider + "/")

        # Test model_name extraction
        model_name = model.model_name()
        assert isinstance(model_name, str)
        assert model_name in model.model_id
        assert model.model_id == f"{provider}/{model_name}"

        # Test cost calculation
        avg_cost = model.cost_per_1m_tokens()
        assert isinstance(avg_cost, float)
        assert avg_cost > 0
        # Average should be between input and output costs
        assert (
            min(model.cost_per_1m_input_tokens, model.cost_per_1m_output_tokens)
            <= avg_cost
            <= max(model.cost_per_1m_input_tokens, model.cost_per_1m_output_tokens)
        )

    def test_all_model_features(self, sample_checkpoint_json: str):
        """Test accessing all ModelFeatures in checkpoint."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Test all models
        for model in checkpoint.models:
            assert isinstance(model.model_id, str)
            assert isinstance(model.error_rates, list) or hasattr(model.error_rates, "__iter__")
            assert len(list(model.error_rates)) == checkpoint.n_clusters
            assert isinstance(model.cost_per_1m_input_tokens, float)
            assert isinstance(model.cost_per_1m_output_tokens, float)


class TestTrainingMetrics:
    """Test TrainingMetrics type bindings."""

    def test_training_metrics_from_checkpoint(self, sample_checkpoint_json: str):
        """Test accessing TrainingMetrics from checkpoint."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        metrics = checkpoint.metrics

        # Test optional fields
        assert hasattr(metrics, "silhouette_score")
        assert hasattr(metrics, "n_samples")
        assert hasattr(metrics, "cluster_sizes")
        assert hasattr(metrics, "inertia")

        # Test silhouette_score access
        score = metrics.silhouette_score
        assert score is not None
        assert isinstance(score, float)
        assert score == pytest.approx(0.85)
        assert -1.0 <= score <= 1.0

    def test_training_metrics_optional_fields(self, sample_checkpoint_json: str):
        """Test accessing optional TrainingMetrics fields."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        metrics = checkpoint.metrics

        # n_samples is optional
        n_samples = metrics.n_samples
        # Can be None or int
        if n_samples is not None:
            assert isinstance(n_samples, int)
            assert n_samples > 0

        # inertia is optional
        inertia = metrics.inertia
        # Can be None or float
        if inertia is not None:
            assert isinstance(inertia, float)
            assert inertia >= 0.0

    def test_training_metrics_cluster_sizes(self, sample_checkpoint_json: str):
        """Test accessing cluster_sizes vector from TrainingMetrics."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        metrics = checkpoint.metrics

        # cluster_sizes is optional, so it might be None
        cluster_sizes = metrics.cluster_sizes
        # If present, it should be a list or iterable - this will fail if vector.h is missing
        if cluster_sizes is not None:
            cluster_sizes_list = list(cluster_sizes)
            assert isinstance(cluster_sizes_list, list)
            assert all(isinstance(size, int) for size in cluster_sizes_list)
            assert all(size > 0 for size in cluster_sizes_list)
            # Should match number of clusters
            assert len(cluster_sizes_list) == checkpoint.n_clusters


class TestEmbeddingConfig:
    """Test EmbeddingConfig type bindings."""

    def test_embedding_config_properties(self, sample_checkpoint_json: str):
        """Test EmbeddingConfig property access."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        config = checkpoint.embedding

        assert isinstance(config.model, str)
        assert len(config.model) > 0
        assert isinstance(config.trust_remote_code, bool)

    def test_embedding_config_consistency(self, sample_checkpoint_json: str):
        """Test EmbeddingConfig consistency with checkpoint."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        config = checkpoint.embedding

        # Checkpoint convenience accessors should match
        assert checkpoint.embedding_model == config.model
        assert checkpoint.allow_trust_remote_code == config.trust_remote_code


class TestClusteringConfig:
    """Test ClusteringConfig type bindings."""

    def test_clustering_config_properties(self, sample_checkpoint_json: str):
        """Test ClusteringConfig property access."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        config = checkpoint.clustering

        assert isinstance(config.n_clusters, int)
        assert config.n_clusters > 0
        assert isinstance(config.random_state, int)
        assert isinstance(config.max_iter, int)
        assert config.max_iter > 0
        assert isinstance(config.n_init, int)
        assert config.n_init > 0
        assert isinstance(config.algorithm, str)
        assert len(config.algorithm) > 0
        assert isinstance(config.normalization, str)
        assert len(config.normalization) > 0

    def test_clustering_config_consistency(self, sample_checkpoint_json: str):
        """Test ClusteringConfig consistency with checkpoint."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        config = checkpoint.clustering

        # Checkpoint convenience accessors should match
        assert checkpoint.n_clusters == config.n_clusters
        assert checkpoint.random_state == config.random_state
