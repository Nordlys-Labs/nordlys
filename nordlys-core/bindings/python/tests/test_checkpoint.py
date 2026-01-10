"""Tests for Python bindings - NordlysCheckpoint class."""

import json
from pathlib import Path

import pytest


class TestNordlysCheckpointCreation:
    """Test NordlysCheckpoint factory methods."""

    def test_from_json_string(self, sample_checkpoint_json: str):
        """Test creating checkpoint from JSON string."""
        from nordlys_core_ext import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        assert checkpoint.n_clusters == 3
        assert checkpoint.embedding_model == "test-model"
        assert checkpoint.dtype == "float32"

    def test_from_json_file(self, sample_checkpoint_path: Path):
        """Test creating checkpoint from JSON file."""
        from nordlys_core_ext import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_file(str(sample_checkpoint_path))
        assert checkpoint.n_clusters == 3
        assert checkpoint.dtype == "float32"

    def test_from_msgpack_string(self, sample_checkpoint_json: str):
        """Test creating checkpoint from MessagePack string."""
        from nordlys_core_ext import NordlysCheckpoint

        # First create a checkpoint and serialize to msgpack
        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        msgpack_data = checkpoint.to_msgpack_bytes()

        # Now deserialize from msgpack
        loaded = NordlysCheckpoint.from_msgpack_bytes(msgpack_data)
        assert loaded.n_clusters == 3
        assert loaded.dtype == "float32"

    def test_from_msgpack_file(self, tmp_path: Path, sample_checkpoint_json: str):
        """Test creating checkpoint from MessagePack file."""
        from nordlys_core_ext import NordlysCheckpoint

        # Create checkpoint and write to msgpack file
        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        msgpack_path = tmp_path / "test_checkpoint.msgpack"
        checkpoint.to_msgpack_file(str(msgpack_path))

        # Load from msgpack file
        loaded = NordlysCheckpoint.from_msgpack_file(str(msgpack_path))
        assert loaded.n_clusters == 3
        assert loaded.dtype == "float32"

    def test_invalid_json_raises(self):
        """Test that invalid JSON raises an error."""
        from nordlys_core_ext import NordlysCheckpoint

        with pytest.raises(RuntimeError):
            NordlysCheckpoint.from_json_string("not valid json")

    def test_float64_support(self, sample_checkpoint_json_float64: str):
        """Test float64 checkpoint creation."""
        from nordlys_core_ext import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json_float64)
        assert checkpoint.dtype == "float64"


class TestNordlysCheckpointSerialization:
    """Test NordlysCheckpoint serialization methods."""

    def test_json_round_trip(self, sample_checkpoint_json: str):
        """Test JSON serialization round-trip."""
        from nordlys_core_ext import NordlysCheckpoint

        # Load from JSON
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Serialize to JSON string
        json_str = original.to_json_string()

        # Parse back
        loaded = NordlysCheckpoint.from_json_string(json_str)

        # Verify properties match
        assert loaded.n_clusters == original.n_clusters
        assert loaded.embedding_model == original.embedding_model
        assert loaded.dtype == original.dtype

    def test_msgpack_round_trip(self, sample_checkpoint_json: str):
        """Test MessagePack serialization round-trip."""
        from nordlys_core_ext import NordlysCheckpoint

        # Load from JSON
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Serialize to msgpack
        msgpack_data = original.to_msgpack_bytes()

        # Deserialize from msgpack
        loaded = NordlysCheckpoint.from_msgpack_bytes(msgpack_data)

        # Verify properties match
        assert loaded.n_clusters == original.n_clusters
        assert loaded.embedding_model == original.embedding_model
        assert loaded.dtype == original.dtype

    def test_file_operations_json(self, tmp_path: Path, sample_checkpoint_json: str):
        """Test JSON file operations."""
        from nordlys_core_ext import NordlysCheckpoint

        # Create checkpoint
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Write to file
        json_path = tmp_path / "checkpoint.json"
        original.to_json_file(str(json_path))

        # Read back
        loaded = NordlysCheckpoint.from_json_file(str(json_path))

        assert loaded.n_clusters == original.n_clusters
        assert loaded.dtype == original.dtype

    def test_file_operations_msgpack(self, tmp_path: Path, sample_checkpoint_json: str):
        """Test MessagePack file operations."""
        from nordlys_core_ext import NordlysCheckpoint

        # Create checkpoint
        original = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Write to file
        msgpack_path = tmp_path / "checkpoint.msgpack"
        original.to_msgpack_file(str(msgpack_path))

        # Read back
        loaded = NordlysCheckpoint.from_msgpack_file(str(msgpack_path))

        assert loaded.n_clusters == original.n_clusters
        assert loaded.dtype == original.dtype


class TestNordlysCheckpointValidation:
    """Test NordlysCheckpoint validation."""

    def test_valid_checkpoint_passes_validation(self, sample_checkpoint_json: str):
        """Test that valid checkpoints pass validation."""
        from nordlys_core_ext import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)
        # Should not raise
        checkpoint.validate()

    def test_invalid_checkpoint_fails_validation(self, sample_checkpoint_json: str):
        """Test that invalid checkpoints fail validation during parsing."""
        from nordlys_core_ext import NordlysCheckpoint

        # Create a checkpoint with invalid n_clusters - validation happens during parsing
        invalid_json = json.loads(sample_checkpoint_json)
        invalid_json["clustering"]["n_clusters"] = -1

        with pytest.raises((RuntimeError, ValueError)):
            NordlysCheckpoint.from_json_string(json.dumps(invalid_json))


class TestNordlysCheckpointProperties:
    """Test NordlysCheckpoint property access."""

    def test_properties_accessible(self, sample_checkpoint_json: str):
        """Test that all properties are accessible."""
        from nordlys_core_ext import NordlysCheckpoint

        checkpoint = NordlysCheckpoint.from_json_string(sample_checkpoint_json)

        # Test basic properties
        assert isinstance(checkpoint.n_clusters, int)
        assert isinstance(checkpoint.embedding_model, str)
        assert isinstance(checkpoint.dtype, str)

        # Test silhouette_score
        score = checkpoint.silhouette_score
        assert isinstance(score, float)
        assert score == pytest.approx(0.85)

        # Test dtype consistency
        assert checkpoint.dtype in ("float32", "float64")
