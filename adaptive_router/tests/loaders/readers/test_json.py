"""Tests for JSONProfileReader."""

import json
from pathlib import Path

import pytest

from adaptive_router.loaders.readers.json import JSONProfileReader
from adaptive_router.models.storage import RouterProfile


@pytest.fixture
def valid_profile_data() -> dict:
    """Create valid profile data for testing."""
    """Create valid profile data for testing."""
    return {
        "metadata": {
            "n_clusters": 2,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "silhouette_score": 0.45,
        },
        "cluster_centers": {
            "n_clusters": 2,
            "feature_dim": 2,
            "cluster_centers": [[0.1, 0.2], [0.4, 0.5]],
        },
        "models": [
            {
                "provider": "openai",
                "model_name": "gpt-4",
                "cost_per_1m_input_tokens": 30.0,
                "cost_per_1m_output_tokens": 60.0,
                "error_rates": [0.05, 0.03],
            }
        ],
    }


@pytest.fixture
def sample_profile(valid_profile_data) -> RouterProfile:
    """Create a sample RouterProfile for testing."""
    return RouterProfile(**valid_profile_data)


@pytest.fixture
def json_profile_file(tmp_path, valid_profile_data) -> Path:
    """Create a temporary JSON profile file."""
    file_path = tmp_path / "test_profile.json"
    with open(file_path, "w") as f:
        json.dump(valid_profile_data, f)
    return file_path


class TestJSONProfileReader:
    """Test JSONProfileReader."""

    def test_read_from_path_valid_json(self, json_profile_file, sample_profile):
        """Test reading valid JSON profile from file path."""
        reader = JSONProfileReader()
        profile = reader.read_from_path(json_profile_file)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == sample_profile.metadata.n_clusters
        assert (
            profile.metadata.embedding_model == sample_profile.metadata.embedding_model
        )

    def test_read_from_path_file_not_found(self):
        """Test read_from_path raises FileNotFoundError for missing file."""
        reader = JSONProfileReader()
        with pytest.raises(FileNotFoundError):
            reader.read_from_path(Path("nonexistent.json"))

    def test_read_from_path_invalid_json(self, tmp_path):
        """Test read_from_path raises ValueError for invalid JSON."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{invalid json")

        reader = JSONProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            reader.read_from_path(invalid_file)

    def test_read_from_bytes_valid_json(self, valid_profile_data, sample_profile):
        """Test reading valid JSON profile from bytes."""
        json_bytes = json.dumps(valid_profile_data).encode("utf-8")

        reader = JSONProfileReader()
        profile = reader.read_from_bytes(json_bytes)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == sample_profile.metadata.n_clusters

    def test_read_from_bytes_invalid_json(self):
        """Test read_from_bytes raises ValueError for invalid JSON."""
        reader = JSONProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            reader.read_from_bytes(b"{invalid json")

    def test_read_from_bytes_invalid_utf8(self):
        """Test read_from_bytes raises ValueError for invalid UTF-8."""
        reader = JSONProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            reader.read_from_bytes(b"\xff\xfe\xfd")  # Invalid UTF-8

    def test_supported_extensions(self):
        """Test supported_extensions returns correct extensions."""
        extensions = JSONProfileReader.supported_extensions()
        assert extensions == [".json"]
