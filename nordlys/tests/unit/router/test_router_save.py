"""Tests for Router.checkpoint() and Router.save()."""

import json

import pytest

from nordlys import Router


class TestRouterCheckpointAccessor:
    """Tests for Router.checkpoint()."""

    def test_checkpoint_returns_checkpoint_object(self, sample_checkpoint):
        """checkpoint() returns the NordlysCheckpoint object."""
        router = Router(checkpoint=sample_checkpoint)
        cp = router.checkpoint()
        assert cp is not None
        assert hasattr(cp, "cluster_centers")
        assert hasattr(cp, "models")
        assert hasattr(cp, "embedding")

    def test_checkpoint_returns_same_object_as_stored(self, sample_checkpoint):
        """checkpoint() returns the same object internally stored."""
        router = Router(checkpoint=sample_checkpoint)
        cp = router.checkpoint()
        assert cp is router._checkpoint

    def test_checkpoint_reloadable(self, tmp_path, sample_checkpoint):
        """Router checkpoint round-trips: save -> reload -> same models/clusters."""
        router1 = Router(checkpoint=sample_checkpoint)

        saved_path = tmp_path / "router_checkpoint.json"
        router1.save(saved_path)

        router2 = Router(checkpoint=str(saved_path))

        assert router2._nr_clusters == router1._nr_clusters
        assert router2._model_ids == router1._model_ids
        cp = router2.checkpoint()
        assert cp.cluster_centers is not None
        assert len(cp.cluster_centers) == router1._nr_clusters


class TestRouterSave:
    """Tests for Router.save()."""

    def test_save_writes_json_file(self, tmp_path, sample_checkpoint):
        """save() writes a valid JSON checkpoint file."""
        router = Router(checkpoint=sample_checkpoint)
        path = tmp_path / "saved.json"
        result = router.save(path)

        assert result == path
        assert path.exists()
        content = json.loads(path.read_text())
        assert "cluster_centers" in content
        assert "models" in content
        assert "embedding" in content

    def test_save_creates_parent_dirs(self, tmp_path, sample_checkpoint):
        """save() creates missing parent directories."""
        router = Router(checkpoint=sample_checkpoint)
        path = tmp_path / "subdir" / "nested" / "checkpoint.json"
        router.save(path)

        assert path.exists()

    def test_save_returns_resolved_path(self, tmp_path, sample_checkpoint):
        """save() returns the resolved Path."""
        router = Router(checkpoint=sample_checkpoint)
        relative = tmp_path / "checkpoint.json"
        result = router.save(str(relative))

        assert result == relative.resolve()

    def test_save_overwrites_existing(self, tmp_path, sample_checkpoint):
        """save() overwrites an existing checkpoint file."""
        router = Router(checkpoint=sample_checkpoint)
        path = tmp_path / "checkpoint.json"
        path.write_text("dummy")

        router.save(path)

        content = json.loads(path.read_text())
        assert "cluster_centers" in content

    def test_saved_checkpoint_loads_and_routes(self, tmp_path, sample_checkpoint):
        """Saved checkpoint round-trips: save -> reload -> same clusters and models."""
        router1 = Router(checkpoint=sample_checkpoint)

        path = tmp_path / "router.json"
        router1.save(path)

        router2 = Router(checkpoint=str(path))

        assert router2._nr_clusters == router1._nr_clusters
        assert router2._model_ids == router1._model_ids
