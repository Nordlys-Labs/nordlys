"""Tests for JSONProfileWriter."""

import json

import pytest

from adaptive_router.models.storage import RouterProfile
from adaptive_router.savers.writers.json import JSONProfileWriter


@pytest.fixture
def valid_profile_data() -> dict:
    """Create valid profile data for testing."""
    # Full RouterProfile structure with all required fields
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


class TestJSONProfileWriter:
    """Test JSONProfileWriter."""

    def test_write_to_path_valid_profile(self, tmp_path, sample_profile):
        """Test writing profile to JSON file path."""
        writer = JSONProfileWriter()
        output_path = tmp_path / "test_profile.json"

        writer.write_to_path(sample_profile, output_path)

        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = json.load(f)
        assert data["metadata"]["n_clusters"] == 2

    def test_write_to_path_creates_directories(self, tmp_path, sample_profile):
        """Test writing profile creates parent directories."""
        writer = JSONProfileWriter()
        output_path = tmp_path / "subdir" / "nested" / "test_profile.json"

        writer.write_to_path(sample_profile, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_write_to_bytes_valid_profile(self, sample_profile):
        """Test writing profile to JSON bytes."""
        writer = JSONProfileWriter()
        data = writer.write_to_bytes(sample_profile)

        assert isinstance(data, bytes)

        # Verify content
        parsed_data = json.loads(data.decode("utf-8"))
        assert parsed_data["metadata"]["n_clusters"] == 2

    def test_supported_extensions(self):
        """Test supported_extensions returns correct extensions."""
        extensions = JSONProfileWriter.supported_extensions()
        assert extensions == [".json"]
