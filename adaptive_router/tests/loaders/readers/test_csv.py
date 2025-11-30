"""Tests for CSVProfileReader."""

import csv
import json
from pathlib import Path

import pytest

from adaptive_router.loaders.readers.csv import CSVProfileReader
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
def csv_profile_file(tmp_path, valid_profile_data) -> Path:
    """Create a temporary CSV profile file."""
    file_path = tmp_path / "test_profile.csv"
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["format", "data"])
        writer.writeheader()
        writer.writerow({"format": "json", "data": json.dumps(valid_profile_data)})
    return file_path


class TestCSVProfileReader:
    """Test CSVProfileReader."""

    def test_read_from_path_valid_csv(self, csv_profile_file, sample_profile):
        """Test reading valid CSV profile from file path."""
        reader = CSVProfileReader()
        profile = reader.read_from_path(csv_profile_file)

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == sample_profile.metadata.n_clusters
        assert (
            profile.metadata.embedding_model == sample_profile.metadata.embedding_model
        )

    def test_read_from_path_file_not_found(self):
        """Test read_from_path raises FileNotFoundError for missing file."""
        reader = CSVProfileReader()
        with pytest.raises(FileNotFoundError):
            reader.read_from_path(Path("nonexistent.csv"))

    def test_read_from_path_empty_csv(self, tmp_path):
        """Test read_from_path raises ValueError for empty CSV file."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("format,data\n")  # Header only, no data rows

        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="CSV file.*is empty"):
            reader.read_from_path(empty_file)

    def test_read_from_path_missing_format_column(self, tmp_path):
        """Test read_from_path raises ValueError for missing format column."""
        invalid_file = tmp_path / "invalid.csv"
        with open(invalid_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["data"])
            writer.writeheader()
            writer.writerow({"data": "some data"})

        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="missing 'format' column"):
            reader.read_from_path(invalid_file)

    def test_read_from_path_wrong_format_value(self, tmp_path):
        """Test read_from_path raises ValueError for wrong format value."""
        invalid_file = tmp_path / "invalid.csv"
        with open(invalid_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["format", "data"])
            writer.writeheader()
            writer.writerow({"format": "xml", "data": "some data"})

        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read_from_path(invalid_file)

    def test_read_from_path_missing_data_column(self, tmp_path):
        """Test read_from_path raises ValueError for missing data column."""
        invalid_file = tmp_path / "invalid.csv"
        with open(invalid_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["format"])
            writer.writeheader()
            writer.writerow({"format": "json"})

        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="missing 'data' column"):
            reader.read_from_path(invalid_file)

    def test_read_from_path_invalid_json_in_data(self, tmp_path):
        """Test read_from_path raises ValueError for invalid JSON in data column."""
        invalid_file = tmp_path / "invalid.csv"
        with open(invalid_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["format", "data"])
            writer.writeheader()
            writer.writerow({"format": "json", "data": "{invalid json"})

        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="Invalid JSON data"):
            reader.read_from_path(invalid_file)

    def test_read_from_bytes_valid_csv(self, valid_profile_data, sample_profile):
        """Test reading valid CSV profile from bytes."""
        # Create CSV content
        import io

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["format", "data"])
        writer.writeheader()
        writer.writerow({"format": "json", "data": json.dumps(valid_profile_data)})
        csv_content = output.getvalue()

        reader = CSVProfileReader()
        profile = reader.read_from_bytes(csv_content.encode("utf-8"))

        assert isinstance(profile, RouterProfile)
        assert profile.metadata.n_clusters == sample_profile.metadata.n_clusters

    def test_read_from_bytes_empty_csv(self):
        """Test read_from_bytes raises ValueError for empty CSV data."""
        # CSV with header only
        csv_content = "format,data\n"
        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="CSV data is empty"):
            reader.read_from_bytes(csv_content.encode("utf-8"))

    def test_read_from_bytes_missing_format_column(self):
        """Test read_from_bytes raises ValueError for missing format column."""
        csv_content = "data\nsome data\n"
        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="missing 'format' column"):
            reader.read_from_bytes(csv_content.encode("utf-8"))

    def test_read_from_bytes_wrong_format_value(self):
        """Test read_from_bytes raises ValueError for wrong format value."""
        csv_content = "format,data\nxml,some data\n"
        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="Unsupported format"):
            reader.read_from_bytes(csv_content.encode("utf-8"))

    def test_read_from_bytes_missing_data_column(self):
        """Test read_from_bytes raises ValueError for missing data column."""
        csv_content = "format\njson\n"
        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="missing 'data' column"):
            reader.read_from_bytes(csv_content.encode("utf-8"))

    def test_read_from_bytes_invalid_json_in_data(self):
        """Test read_from_bytes raises ValueError for invalid JSON in data column."""
        csv_content = "format,data\njson,{invalid json\n"
        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="Invalid data in CSV"):
            reader.read_from_bytes(csv_content.encode("utf-8"))

    def test_read_from_bytes_invalid_utf8(self):
        """Test read_from_bytes raises ValueError for invalid UTF-8."""
        reader = CSVProfileReader()
        with pytest.raises(ValueError, match="Invalid data in CSV"):
            reader.read_from_bytes(b"\xff\xfe\xfd")  # Invalid UTF-8

    def test_supported_extensions(self):
        """Test supported_extensions returns correct extensions."""
        extensions = CSVProfileReader.supported_extensions()
        assert extensions == [".csv"]
