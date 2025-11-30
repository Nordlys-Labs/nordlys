"""Tests for ParquetProfileReader."""

import json
from pathlib import Path

import polars as pl
import pytest

from adaptive_router.loaders.readers.parquet import ParquetProfileReader
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
def parquet_profile_file(tmp_path, valid_profile_data) -> Path:
    """Create a temporary Parquet profile file."""
    file_path = tmp_path / "test_profile.parquet"
    df = pl.DataFrame([{"format": "json", "data": json.dumps(valid_profile_data)}])
    df.write_parquet(file_path)
    return file_path


class TestParquetProfileReader:
    """Test ParquetProfileReader."""

    def test_read_from_path_valid_parquet(self, parquet_profile_file, sample_profile):
        """Test reading valid Parquet profile from file path."""
        reader = ParquetProfileReader()
        profile = reader.read_from_path(parquet_profile_file)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == sample_profile.metadata.n_clusters
        assert (
            profile.metadata.embedding_model == sample_profile.metadata.embedding_model
        )

    def test_read_from_path_file_not_found(self):
        """Test read_from_path raises FileNotFoundError for missing file."""
        reader = ParquetProfileReader()
        with pytest.raises(FileNotFoundError):
            reader.read_from_path(Path("nonexistent.parquet"))

    def test_read_from_path_empty_parquet(self, tmp_path):
        """Test read_from_path raises ValueError for empty Parquet file."""
        empty_file = tmp_path / "empty.parquet"
        df = pl.DataFrame()
        df.write_parquet(empty_file)

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="Parquet file.*is empty"):
            reader.read_from_path(empty_file)

    def test_read_from_path_missing_format_column(self, tmp_path):
        """Test read_from_path raises ValueError for missing format column."""
        invalid_file = tmp_path / "invalid.parquet"
        df = pl.DataFrame([{"data": "some data"}])  # Missing format column
        df.write_parquet(invalid_file)

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="missing 'format' column"):
            reader.read_from_path(invalid_file)

    def test_read_from_path_wrong_format_value(self, tmp_path):
        """Test read_from_path raises ValueError for wrong format value."""
        invalid_file = tmp_path / "invalid.parquet"
        df = pl.DataFrame([{"format": "xml", "data": "some data"}])  # Wrong format
        df.write_parquet(invalid_file)

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read_from_path(invalid_file)

    def test_read_from_path_missing_data_column(self, tmp_path):
        """Test read_from_path raises ValueError for missing data column."""
        invalid_file = tmp_path / "invalid.parquet"
        df = pl.DataFrame([{"format": "json"}])  # Missing data column
        df.write_parquet(invalid_file)

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="missing 'data' column"):
            reader.read_from_path(invalid_file)

    def test_read_from_path_invalid_json_in_data(self, tmp_path):
        """Test read_from_path raises ValueError for invalid JSON in data column."""
        invalid_file = tmp_path / "invalid.parquet"
        df = pl.DataFrame([{"format": "json", "data": "{invalid json"}])
        df.write_parquet(invalid_file)

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON data"):
            reader.read_from_path(invalid_file)

    def test_read_from_bytes_valid_parquet(self, valid_profile_data, sample_profile):
        """Test reading valid Parquet profile from bytes."""
        df = pl.DataFrame([{"format": "json", "data": json.dumps(valid_profile_data)}])
        import io

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        reader = ParquetProfileReader()
        profile = reader.read_from_bytes(parquet_bytes)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == sample_profile.metadata.n_clusters

    def test_read_from_bytes_empty_parquet(self):
        """Test read_from_bytes raises ValueError for empty Parquet data."""
        df = pl.DataFrame()
        import io

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        empty_bytes = buffer.getvalue()

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="Parquet data is empty"):
            reader.read_from_bytes(empty_bytes)

    def test_read_from_bytes_missing_format_column(self):
        """Test read_from_bytes raises ValueError for missing format column."""
        df = pl.DataFrame([{"data": "some data"}])  # Missing format column
        import io

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        invalid_bytes = buffer.getvalue()

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="missing 'format' column"):
            reader.read_from_bytes(invalid_bytes)

    def test_read_from_bytes_wrong_format_value(self):
        """Test read_from_bytes raises ValueError for wrong format value."""
        df = pl.DataFrame([{"format": "xml", "data": "some data"}])  # Wrong format
        import io

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        invalid_bytes = buffer.getvalue()

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read_from_bytes(invalid_bytes)

    def test_read_from_bytes_missing_data_column(self):
        """Test read_from_bytes raises ValueError for missing data column."""
        df = pl.DataFrame([{"format": "json"}])  # Missing data column
        import io

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        invalid_bytes = buffer.getvalue()

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="missing 'data' column"):
            reader.read_from_bytes(invalid_bytes)

    def test_read_from_bytes_invalid_json_in_data(self):
        """Test read_from_bytes raises ValueError for invalid JSON in data column."""
        df = pl.DataFrame([{"format": "json", "data": "{invalid json"}])
        import io

        buffer = io.BytesIO()
        df.write_parquet(buffer)
        invalid_bytes = buffer.getvalue()

        reader = ParquetProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON data"):
            reader.read_from_bytes(invalid_bytes)

    def test_supported_extensions(self):
        """Test supported_extensions returns correct extensions."""
        extensions = ParquetProfileReader.supported_extensions()
        assert extensions == [".parquet", ".pq"]
