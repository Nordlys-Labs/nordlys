"""Tests for LocalFileProfileLoader."""

import json
from pathlib import Path

import polars as pl
import pytest

from adaptive_router.loaders.local import LocalFileProfileLoader


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
def json_profile_file(tmp_path, valid_profile_data) -> Path:
    """Create a temporary JSON profile file."""
    file_path = tmp_path / "test_profile.json"
    with open(file_path, "w") as f:
        json.dump(valid_profile_data, f)
    return file_path


@pytest.fixture
def csv_profile_file(tmp_path, valid_profile_data) -> Path:
    """Create a temporary CSV profile file."""
    file_path = tmp_path / "test_profile.csv"
    with open(file_path, "w", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=["format", "data"])
        writer.writeheader()
        writer.writerow({"format": "json", "data": json.dumps(valid_profile_data)})
    return file_path


@pytest.fixture
def parquet_profile_file(tmp_path, valid_profile_data) -> Path:
    """Create a temporary Parquet profile file."""
    file_path = tmp_path / "test_profile.parquet"
    df = pl.DataFrame([{"format": "json", "data": json.dumps(valid_profile_data)}])
    df.write_parquet(file_path)
    return file_path


class TestLocalFileProfileLoaderInitialization:
    """Test LocalFileProfileLoader initialization."""

    def test_initialization_with_valid_path(self, json_profile_file: Path) -> None:
        """Test loader initializes with valid file path."""
        loader = LocalFileProfileLoader(json_profile_file)
        assert loader.profile_path == json_profile_file

    def test_initialization_with_string_path(self, json_profile_file: Path) -> None:
        """Test loader accepts string path."""
        loader = LocalFileProfileLoader(str(json_profile_file))
        assert loader.profile_path == json_profile_file

    def test_initialization_with_nonexistent_path(self, tmp_path) -> None:
        """Test loader raises error for nonexistent file."""
        nonexistent = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Profile file not found"):
            LocalFileProfileLoader(nonexistent)


class TestLocalFileProfileLoaderLoadProfile:
    """Test LocalFileProfileLoader load_profile method."""

    # Note: Full RouterProfile schema is complex (requires cluster_centers, models with error_rates,
    # and metadata with specific nested structures).
    # We test error paths which are more critical for robustness.

    def test_load_profile_from_csv(self, csv_profile_file) -> None:
        """Test loading profile from CSV file."""
        loader = LocalFileProfileLoader(csv_profile_file)
        profile = loader.load_profile()

        assert profile.metadata.n_clusters == 2
        assert len(profile.models) == 1

    def test_load_profile_from_parquet(self, parquet_profile_file) -> None:
        """Test loading profile from Parquet file."""
        loader = LocalFileProfileLoader(parquet_profile_file)
        profile = loader.load_profile()

        assert profile.metadata.n_clusters == 2
        assert len(profile.models) == 1

    def test_load_profile_with_corrupted_json(self, tmp_path) -> None:
        """Test loading corrupted JSON raises ValueError."""
        corrupted_file = tmp_path / "corrupted.json"
        with open(corrupted_file, "w") as f:
            f.write("{invalid json content")

        loader = LocalFileProfileLoader(corrupted_file)
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_profile()

    def test_load_profile_with_invalid_schema(self, tmp_path) -> None:
        """Test loading profile with invalid schema raises ValueError."""
        invalid_file = tmp_path / "invalid.json"
        with open(invalid_file, "w") as f:
            json.dump({"invalid": "schema"}, f)

        loader = LocalFileProfileLoader(invalid_file)
        with pytest.raises(ValueError, match="validation"):
            loader.load_profile()


class TestLocalFileProfileLoaderHealthCheck:
    """Test LocalFileProfileLoader health_check method."""

    def test_health_check_with_valid_file(self, json_profile_file: Path) -> None:
        """Test health check returns True for valid file."""
        loader = LocalFileProfileLoader(json_profile_file)
        assert loader.health_check() is True

    def test_health_check_after_file_deletion(self, json_profile_file: Path) -> None:
        """Test health check returns False after file deletion."""
        loader = LocalFileProfileLoader(json_profile_file)
        assert loader.health_check() is True

        json_profile_file.unlink()
        assert loader.health_check() is False

    def test_health_check_with_directory_path(self, tmp_path) -> None:
        """Test health check returns False for directory."""
        # Create a directory with the profile name
        dir_path = tmp_path / "profile.json"
        dir_path.mkdir()

        # Create a dummy file to allow initialization
        dummy_file = tmp_path / "dummy.json"
        with open(dummy_file, "w") as f:
            json.dump({"test": "data"}, f)

        loader = LocalFileProfileLoader(dummy_file)
        loader.profile_path = dir_path  # Override with directory

        assert loader.health_check() is False
