"""Integration tests for runtime-only Router behavior."""

import pytest

from nordlys import Router


class TestRouterRuntimeOnly:
    def test_router_has_no_training_methods(self):
        with pytest.raises(AttributeError):
            Router.fit  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            Router.fit_transform  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            Router.transform  # type: ignore[attr-defined]

    def test_route_before_load_raises(self, three_models):
        router = Router(models=three_models)
        with pytest.raises(RuntimeError, match="runtime is not initialized"):
            router.route("test")

    def test_route_batch_before_load_raises(self, three_models):
        router = Router(models=three_models)
        with pytest.raises(RuntimeError, match="runtime is not initialized"):
            router.route_batch(["a", "b"])


class TestRouterRuntimeAttributes:
    def test_centroids_property(self, fitted_nordlys):
        centroids = fitted_nordlys.centroids_
        assert centroids.ndim == 2
        assert centroids.shape[0] > 0

    def test_metrics_property(self, fitted_nordlys):
        metrics = fitted_nordlys.metrics_
        assert metrics.n_clusters > 0
        assert metrics.n_samples is not None

    def test_model_accuracies_property(self, fitted_nordlys):
        model_accuracies = fitted_nordlys.model_accuracies_
        assert isinstance(model_accuracies, dict)
        assert len(model_accuracies) == fitted_nordlys.n_clusters_

    def test_get_cluster_info_uses_checkpoint_sizes(self, fitted_nordlys):
        cluster = fitted_nordlys.get_cluster_info(0)
        assert cluster.cluster_id == 0
        assert cluster.size >= 0
