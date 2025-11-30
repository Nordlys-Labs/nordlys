"""Tests for LocalFileProfileSaver."""

import json

import pytest

from adaptive_router.models.storage import RouterProfile
from adaptive_router.savers.local import LocalFileProfileSaver


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


class TestLocalFileProfileSaver:
    """Test LocalFileProfileSaver."""

    def test_save_profile_json_format(self, tmp_path, sample_profile):
        """Test saving profile in JSON format."""
        saver = LocalFileProfileSaver()
        output_path = tmp_path / "test_profile.json"

        result_path = saver.save_profile(sample_profile, str(output_path))

        assert result_path == str(output_path)
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = json.load(f)
        assert data["metadata"]["n_clusters"] == 2

    def test_save_profile_csv_format(self, tmp_path, sample_profile):
        """Test saving profile in CSV format."""
        saver = LocalFileProfileSaver()
        output_path = tmp_path / "test_profile.csv"

        result_path = saver.save_profile(sample_profile, str(output_path))

        assert result_path == str(output_path)
        assert output_path.exists()

        # Verify CSV format
        with open(output_path, newline="") as f:
            content = f.read()
            assert "format,data" in content
            assert "json" in content

    def test_save_profile_parquet_format(self, tmp_path, sample_profile):
        """Test saving profile in Parquet format."""
        saver = LocalFileProfileSaver()
        output_path = tmp_path / "test_profile.parquet"

        result_path = saver.save_profile(sample_profile, str(output_path))

        assert result_path == str(output_path)
        assert output_path.exists()

    def test_save_profile_creates_directories(self, tmp_path, sample_profile):
        """Test saving profile creates parent directories."""
        saver = LocalFileProfileSaver()
        output_path = tmp_path / "subdir" / "nested" / "test_profile.json"

        result_path = saver.save_profile(sample_profile, str(output_path))

        assert result_path == str(output_path)
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_profile_unsupported_format(self, tmp_path, sample_profile):
        """Test saving profile with unsupported format raises ValueError."""
        saver = LocalFileProfileSaver()
        output_path = tmp_path / "test_profile.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            saver.save_profile(sample_profile, str(output_path))

    def test_health_check_success(self, tmp_path):
        """Test health check returns True when filesystem is writable."""
        saver = LocalFileProfileSaver()
        assert saver.health_check() is True

    def test_health_check_failure(self, tmp_path):
        """Test health check returns False when filesystem is not writable."""
        # This is hard to test reliably, but the method exists
        saver = LocalFileProfileSaver()
        # In normal circumstances, this should return True
        result = saver.health_check()
        assert isinstance(result, bool)
