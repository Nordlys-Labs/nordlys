"""Tests for Python bindings - NordlysCheckpoint class."""

import json
from pathlib import Path

import pytest


class TestNordlysCheckpointCreation:
    """Test NordlysCheckpoint factory methods."""

    def test_from_json_string(self, sample_checkpoint_json: str):
        """Test creating checkpoint from JSON string."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        assert checkpoint.n_clusters == 3
        assert checkpoint.embedding_model == "test-model"

    def test_from_json_file(self, sample_checkpoint_path: Path):
        """Test creating checkpoint from JSON file."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_file(str(sample_checkpoint_path))
        assert checkpoint.n_clusters == 3

    def test_from_msgpack_string(self, sample_checkpoint_json: str):
        """Test creating checkpoint from MessagePack string."""
        from nordlys_core import NordlysCheckpoint

        # First create a checkpoint and serialize to msgpack
        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        msgpack_data = checkpoint.to_msgpack_bytes()

        # Now deserialize from msgpack
        loaded = NordlysCheckpoint.from_msgpack_bytes(msgpack_data)
        assert loaded.n_clusters == 3

    def test_from_msgpack_file(self, tmp_path: Path, sample_checkpoint_json: str):
        """Test creating checkpoint from MessagePack file."""
        from nordlys_core import NordlysCheckpoint

        # Create checkpoint and write to msgpack file
        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        msgpack_path = tmp_path / "test_checkpoint.msgpack"
        checkpoint.to_msgpack_file(str(msgpack_path))

        # Load from msgpack file
        loaded = NordlysCheckpoint.from_msgpack_file(str(msgpack_path))
        assert loaded.n_clusters == 3

    def test_invalid_json_raises(self):
        """Test that invalid JSON raises an error."""
        from nordlys_core import NordlysCheckpoint

        with pytest.raises(RuntimeError):
            NordlysCheckpoint.from_json_string("not valid json")



class TestNordlysCheckpointSerialization:
    """Test NordlysCheckpoint serialization methods."""

    def test_json_round_trip(self, sample_checkpoint_json: str):
        """Test JSON serialization round-trip."""
        from nordlys_core import NordlysCheckpoint

        # Load from JSON
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Serialize to JSON string
        json_str = original.to_json_string()

        # Parse back
        loaded = NordlysCheckpoint.from_json_string(json_str)

        # Verify properties match
        assert loaded.n_clusters == original.n_clusters
        assert loaded.embedding_model == original.embedding_model

    def test_msgpack_round_trip(self, sample_checkpoint_json: str):
        """Test MessagePack serialization round-trip."""
        from nordlys_core import NordlysCheckpoint

        # Load from JSON
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Serialize to msgpack
        msgpack_data = original.to_msgpack_bytes()

        # Deserialize from msgpack
        loaded = NordlysCheckpoint.from_msgpack_bytes(msgpack_data)

        # Verify properties match
        assert loaded.n_clusters == original.n_clusters
        assert loaded.embedding_model == original.embedding_model

    def test_file_operations_json(self, tmp_path: Path, sample_checkpoint_json: str):
        """Test JSON file operations."""
        from nordlys_core import NordlysCheckpoint

        # Create checkpoint
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Write to file
        json_path = tmp_path / "checkpoint.json"
        original.to_json_file(str(json_path))

        # Read back
        loaded = NordlysCheckpoint.from_json_file(str(json_path))

        assert loaded.n_clusters == original.n_clusters

    def test_file_operations_msgpack(self, tmp_path: Path, sample_checkpoint_json: str):
        """Test MessagePack file operations."""
        from nordlys_core import NordlysCheckpoint

        # Create checkpoint
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Write to file
        msgpack_path = tmp_path / "checkpoint.msgpack"
        original.to_msgpack_file(str(msgpack_path))

        # Read back
        loaded = NordlysCheckpoint.from_msgpack_file(str(msgpack_path))

        assert loaded.n_clusters == original.n_clusters


class TestNordlysCheckpointValidation:
    """Test NordlysCheckpoint validation."""

    def test_valid_checkpoint_passes_validation(self, sample_checkpoint_json: str):
        """Test that valid checkpoints pass validation."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        # Should not raise
        checkpoint.validate()

    def test_invalid_checkpoint_fails_validation(self, sample_checkpoint_json: str):
        """Test that invalid checkpoints fail validation during parsing."""
        from nordlys_core import NordlysCheckpoint

        # Create a checkpoint with invalid n_clusters - validation happens during parsing
        invalid_json = json.loads(sample_checkpoint_json)
        invalid_json["clustering"]["n_clusters"] = -1

        with pytest.raises((RuntimeError, ValueError)):
            NordlysCheckpoint.from_json_string(json.dumps(invalid_json))


class TestNordlysCheckpointProperties:
    """Test NordlysCheckpoint property access."""

    def test_properties_accessible(self, sample_checkpoint_json: str):
        """Test that all properties are accessible."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Test basic properties
        assert isinstance(checkpoint.n_clusters, int)
        assert isinstance(checkpoint.embedding_model, str)
        # Test silhouette_score
        score = checkpoint.silhouette_score
        assert isinstance(score, float)
        assert score == pytest.approx(0.85)

    def test_models_property(self, sample_checkpoint_json: str):
        """Test accessing checkpoint.models property."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Access models - this will fail if vector.h is missing
        models = checkpoint.models
        assert len(models) == 2

        # Iterate over models to get model IDs
        model_ids = [model.model_id for model in models]
        assert "openai/gpt-4" in model_ids
        assert "anthropic/claude-3" in model_ids

        # Access individual model properties
        for model in models:
            assert hasattr(model, "model_id")
            assert hasattr(model, "error_rates")
            assert hasattr(model, "cost_per_1m_input_tokens")
            assert hasattr(model, "cost_per_1m_output_tokens")

    def test_model_features_properties(self, sample_checkpoint_json: str):
        """Test accessing ModelFeatures properties including error_rates vector."""
        from nordlys_core import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Get a model and access its properties
        model = checkpoint.models[0]
        assert isinstance(model.model_id, str)

        # Access error_rates - this will fail if vector.h is missing
        error_rates = model.error_rates
        assert len(error_rates) == checkpoint.n_clusters
        assert all(isinstance(rate, float) for rate in error_rates)
        assert all(0.0 <= rate <= 1.0 for rate in error_rates)
