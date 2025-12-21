"""Tests for Python bindings - RouterProfile class."""

import json
from pathlib import Path

import pytest


class TestRouterProfileCreation:
    """Test RouterProfile factory methods."""

    def test_from_json_string(self, sample_profile_json: str):
        """Test creating profile from JSON string."""
        from adaptive_core_ext import RouterProfile

        profile = RouterProfile.from_json_string(sample_profile_json)
        assert profile.n_clusters == 3
        assert profile.embedding_model == "test-model"
        assert profile.dtype == "float32"
        assert profile.is_float32 is True
        assert profile.is_float64 is False

    def test_from_json_file(self, sample_profile_path: Path):
        """Test creating profile from JSON file."""
        from adaptive_core_ext import RouterProfile

        profile = RouterProfile.from_json_file(str(sample_profile_path))
        assert profile.n_clusters == 3
        assert profile.is_float32 is True

    def test_from_msgpack_string(self, sample_profile_json: str):
        """Test creating profile from MessagePack string."""
        from adaptive_core_ext import RouterProfile

        # First create a profile and serialize to msgpack
        profile = RouterProfile.from_json_string(sample_profile_json)
        msgpack_data = profile.to_msgpack_bytes()

        # Now deserialize from msgpack
        loaded = RouterProfile.from_msgpack_bytes(msgpack_data)
        assert loaded.n_clusters == 3
        assert loaded.is_float32 is True

    def test_from_msgpack_file(self, tmp_path: Path, sample_profile_json: str):
        """Test creating profile from MessagePack file."""
        from adaptive_core_ext import RouterProfile

        # Create profile and write to msgpack file
        profile = RouterProfile.from_json_string(sample_profile_json)
        msgpack_path = tmp_path / "test_profile.msgpack"
        profile.to_msgpack_file(str(msgpack_path))

        # Load from msgpack file
        loaded = RouterProfile.from_msgpack_file(str(msgpack_path))
        assert loaded.n_clusters == 3
        assert loaded.is_float32 is True

    def test_invalid_json_raises(self):
        """Test that invalid JSON raises an error."""
        from adaptive_core_ext import RouterProfile

        with pytest.raises(RuntimeError):
            RouterProfile.from_json_string("not valid json")

    def test_float64_support(self, sample_profile_json_float64: str):
        """Test float64 profile creation."""
        from adaptive_core_ext import RouterProfile

        profile = RouterProfile.from_json_string(sample_profile_json_float64)
        assert profile.dtype == "float64"
        assert profile.is_float64 is True
        assert profile.is_float32 is False


class TestRouterProfileSerialization:
    """Test RouterProfile serialization methods."""

    def test_json_round_trip(self, sample_profile_json: str):
        """Test JSON serialization round-trip."""
        from adaptive_core_ext import RouterProfile

        # Load from JSON
        original = RouterProfile.from_json_string(sample_profile_json)

        # Serialize to JSON string
        json_str = original.to_json_string()

        # Parse back
        loaded = RouterProfile.from_json_string(json_str)

        # Verify properties match
        assert loaded.n_clusters == original.n_clusters
        assert loaded.embedding_model == original.embedding_model
        assert loaded.dtype == original.dtype
        assert loaded.is_float32 == original.is_float32

    def test_msgpack_round_trip(self, sample_profile_json: str):
        """Test MessagePack serialization round-trip."""
        from adaptive_core_ext import RouterProfile

        # Load from JSON
        original = RouterProfile.from_json_string(sample_profile_json)

        # Serialize to msgpack
        msgpack_data = original.to_msgpack_bytes()

        # Deserialize from msgpack
        loaded = RouterProfile.from_msgpack_bytes(msgpack_data)

        # Verify properties match
        assert loaded.n_clusters == original.n_clusters
        assert loaded.embedding_model == original.embedding_model
        assert loaded.dtype == original.dtype
        assert loaded.is_float32 == original.is_float32

    def test_file_operations_json(self, tmp_path: Path, sample_profile_json: str):
        """Test JSON file operations."""
        from adaptive_core_ext import RouterProfile

        # Create profile
        original = RouterProfile.from_json_string(sample_profile_json)

        # Write to file
        json_path = tmp_path / "profile.json"
        original.to_json_file(str(json_path))

        # Read back
        loaded = RouterProfile.from_json_file(str(json_path))

        assert loaded.n_clusters == original.n_clusters
        assert loaded.is_float32 == original.is_float32

    def test_file_operations_msgpack(self, tmp_path: Path, sample_profile_json: str):
        """Test MessagePack file operations."""
        from adaptive_core_ext import RouterProfile

        # Create profile
        original = RouterProfile.from_json_string(sample_profile_json)

        # Write to file
        msgpack_path = tmp_path / "profile.msgpack"
        original.to_msgpack_file(str(msgpack_path))

        # Read back
        loaded = RouterProfile.from_msgpack_file(str(msgpack_path))

        assert loaded.n_clusters == original.n_clusters
        assert loaded.is_float32 == original.is_float32


class TestRouterProfileValidation:
    """Test RouterProfile validation."""

    def test_valid_profile_passes_validation(self, sample_profile_json: str):
        """Test that valid profiles pass validation."""
        from adaptive_core_ext import RouterProfile

        profile = RouterProfile.from_json_string(sample_profile_json)
        # Should not raise
        profile.validate()

    def test_invalid_profile_fails_validation(self, sample_profile_json: str):
        """Test that invalid profiles fail validation."""
        from adaptive_core_ext import RouterProfile

        # Create a profile with invalid data (e.g., negative n_clusters)
        invalid_json = json.loads(sample_profile_json)
        invalid_json["metadata"]["n_clusters"] = -1

        profile = RouterProfile.from_json_string(json.dumps(invalid_json))

        with pytest.raises(ValueError):
            profile.validate()


class TestRouterProfileProperties:
    """Test RouterProfile property access."""

    def test_properties_accessible(self, sample_profile_json: str):
        """Test that all properties are accessible."""
        from adaptive_core_ext import RouterProfile

        profile = RouterProfile.from_json_string(sample_profile_json)

        # Test basic properties
        assert isinstance(profile.n_clusters, int)
        assert isinstance(profile.embedding_model, str)
        assert isinstance(profile.dtype, str)
        assert isinstance(profile.silhouette_score, float)
        assert isinstance(profile.is_float32, bool)
        assert isinstance(profile.is_float64, bool)

        # Test consistency
        if profile.dtype == "float32":
            assert profile.is_float32 is True
            assert profile.is_float64 is False
        elif profile.dtype == "float64":
            assert profile.is_float64 is True
            assert profile.is_float32 is False
