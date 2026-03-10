"""Integration tests for Router persistence (save/load)."""

import json

import pytest

from nordlys import Router


class TestSaveLoad:
    """Test Router save and load methods."""

    def test_save_json(self, fitted_nordlys, tmp_path):
        """Test saving to JSON format."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        assert save_path.exists()
        assert save_path.suffix == ".json"

    def test_save_msgpack(self, fitted_nordlys, tmp_path):
        """Test saving to MessagePack format."""
        save_path = tmp_path / "nordlys.msgpack"
        fitted_nordlys.save(save_path)

        assert save_path.exists()
        assert save_path.suffix == ".msgpack"

    def test_load_json(self, fitted_nordlys, tmp_path):
        """Test loading from JSON format."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        loaded = Router(checkpoint=save_path)

        assert loaded is not None
        assert loaded._is_fitted is True

    def test_load_msgpack(self, fitted_nordlys, tmp_path):
        """Test loading from MessagePack format."""
        save_path = tmp_path / "nordlys.msgpack"
        fitted_nordlys.save(save_path)

        loaded = Router(checkpoint=save_path)

        assert loaded is not None
        assert loaded._is_fitted is True

    def test_loaded_nordlys_is_fitted(self, fitted_nordlys, tmp_path):
        """Test that loaded Router is in fitted state."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        loaded = Router(checkpoint=save_path)

        assert loaded._is_fitted is True
        assert loaded._core_engine is not None

    def test_loaded_nordlys_routes_identically(self, fitted_nordlys, tmp_path):
        """Test that loaded model routes identically."""
        save_path = tmp_path / "nordlys.json"

        # Route before save
        prompt = "What is machine learning?"
        result_before = fitted_nordlys.route(prompt)

        # Save and load
        fitted_nordlys.save(save_path)
        loaded = Router(checkpoint=save_path)

        # Route after load
        result_after = loaded.route(prompt)

        # Should be identical
        assert result_before.model_id == result_after.model_id
        assert result_before.cluster_id == result_after.cluster_id

    def test_load_without_models_uses_checkpoint_models(self, fitted_nordlys, tmp_path):
        """Test loading without providing models uses checkpoint models."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        loaded = Router(checkpoint=save_path)  # No models argument

        assert loaded is not None
        assert loaded._is_fitted is True
        assert len(loaded._models) == 3


class TestCheckpointFormat:
    """Test checkpoint format structure."""

    def test_checkpoint_contains_version(self, fitted_nordlys, tmp_path):
        """Test that saved checkpoint contains version field."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        with open(save_path) as f:
            data = json.load(f)

        assert "version" in data
        assert data["version"] == "2.0"

    def test_checkpoint_contains_models(self, fitted_nordlys, tmp_path):
        """Test that checkpoint contains models array."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        with open(save_path) as f:
            data = json.load(f)

        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 3

    def test_checkpoint_contains_cluster_centers(self, fitted_nordlys, tmp_path):
        """Test that checkpoint contains cluster centers."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        with open(save_path) as f:
            data = json.load(f)

        assert "cluster_centers" in data
        assert isinstance(data["cluster_centers"], list)

    def test_checkpoint_v2_format(self, fitted_nordlys, tmp_path):
        """Test that checkpoint uses v2.0 format structure."""
        save_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(save_path)

        with open(save_path) as f:
            data = json.load(f)

        # v2.0 format fields
        assert "version" in data
        assert "embedding" in data
        assert "clustering" in data
        assert "metrics" in data


class TestPersistenceErrors:
    """Test error handling in persistence."""

    def test_load_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(Exception):  # FileNotFoundError or similar
            Router(checkpoint="nonexistent_file.json")

    def test_load_corrupted_json_raises(self, tmp_path):
        """Test that loading corrupted JSON raises error."""
        save_path = tmp_path / "corrupted.json"
        save_path.write_text("{ invalid json")

        with pytest.raises(Exception):  # JSON decode error
            Router(checkpoint=save_path)

    def test_save_before_fit_raises(self, three_models, tmp_path):
        """Test that saving before checkpoint load raises RuntimeError."""
        nordlys = Router(models=three_models)
        save_path = tmp_path / "nordlys.json"

        with pytest.raises(RuntimeError, match="runtime is not initialized"):
            nordlys.save(save_path)

    def test_save_raises_when_accuracy_missing(self, fitted_nordlys, tmp_path):
        """Test checkpoint creation fails when per-cluster accuracy is missing."""
        model_id = fitted_nordlys._models[0].id
        fitted_nordlys._model_accuracies[0].pop(model_id)

        save_path = tmp_path / "nordlys.json"
        with pytest.raises(ValueError, match="Missing accuracy"):
            fitted_nordlys.save(save_path)


class TestPersistenceConsistency:
    """Test consistency across save/load cycles."""

    def test_multiple_save_load_cycles(self, fitted_nordlys, tmp_path):
        """Test multiple save/load cycles preserve behavior."""
        prompt = "Explain neural networks"

        # Original routing
        result1 = fitted_nordlys.route(prompt)

        # Cycle 1
        path1 = tmp_path / "cycle1.json"
        fitted_nordlys.save(path1)
        loaded1 = Router(checkpoint=path1)
        result2 = loaded1.route(prompt)

        # Cycle 2
        path2 = tmp_path / "cycle2.json"
        loaded1.save(path2)
        loaded2 = Router(checkpoint=path2)
        result3 = loaded2.route(prompt)

        # All should be identical
        assert result1.model_id == result2.model_id == result3.model_id
        assert result1.cluster_id == result2.cluster_id == result3.cluster_id

    def test_json_and_msgpack_equivalent(self, fitted_nordlys, tmp_path):
        """Test that JSON and MessagePack produce equivalent results."""
        prompt = "Test prompt"

        # Save as JSON
        json_path = tmp_path / "nordlys.json"
        fitted_nordlys.save(json_path)
        loaded_json = Router(checkpoint=json_path)
        result_json = loaded_json.route(prompt)

        # Save as MessagePack
        msgpack_path = tmp_path / "nordlys.msgpack"
        fitted_nordlys.save(msgpack_path)
        loaded_msgpack = Router(checkpoint=msgpack_path)
        result_msgpack = loaded_msgpack.route(prompt)

        # Results should be identical
        assert result_json.model_id == result_msgpack.model_id
        assert result_json.cluster_id == result_msgpack.cluster_id
