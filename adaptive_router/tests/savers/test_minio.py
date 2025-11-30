"""Tests for MinIOProfileSaver."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from adaptive_router.models.storage import RouterProfile, MinIOSettings
from adaptive_router.savers.minio import MinIOProfileSaver


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


@pytest.fixture
def minio_settings() -> MinIOSettings:
    """Create sample MinIO settings for testing."""
    return MinIOSettings(
        bucket_name="test-bucket",
        region="us-east-1",
        profile_key="test-profile.json",
        endpoint_url="http://localhost:9000",
        root_user="test-user",
        root_password="test-password",
        connect_timeout=5,
        read_timeout=30,
    )


class TestMinIOProfileSaver:
    """Test MinIOProfileSaver."""

    @patch("adaptive_router.savers.minio.boto3.client")
    def test_init(self, mock_boto3_client):
        """Test MinIOProfileSaver initialization."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        assert saver.bucket_name == "test-bucket"
        assert saver.profile_key == "test-profile.json"
        assert saver.endpoint_url == "http://localhost:9000"

        mock_boto3_client.assert_called_once()
        call_args = mock_boto3_client.call_args
        assert call_args[1]["endpoint_url"] == "http://localhost:9000"
        assert call_args[1]["aws_access_key_id"] == "test-user"
        assert call_args[1]["aws_secret_access_key"] == "test-password"

    @patch("adaptive_router.savers.minio.boto3.client")
    def test_from_settings(self, mock_boto3_client, minio_settings):
        """Test creating saver from MinIOSettings."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        saver = MinIOProfileSaver.from_settings(minio_settings)

        assert saver.bucket_name == "test-bucket"
        assert saver.profile_key == "test-profile.json"
        assert saver.endpoint_url == "http://localhost:9000"

    @patch("adaptive_router.savers.minio.boto3.client")
    @patch("adaptive_router.savers.minio.get_writer")
    def test_save_profile_s3_path(
        self, mock_get_writer, mock_boto3_client, sample_profile
    ):
        """Test saving profile with S3 path format."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        mock_writer = MagicMock()
        mock_writer.write_to_bytes.return_value = b"test data"
        mock_get_writer.return_value = mock_writer

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        result_path = saver.save_profile(
            sample_profile, "s3://my-bucket/my-profile.json"
        )

        assert result_path == "s3://my-bucket/my-profile.json"
        mock_s3.put_object.assert_called_once_with(
            Bucket="my-bucket",
            Key="my-profile.json",
            Body=b"test data",
            ContentType="application/json",
        )

    @patch("adaptive_router.savers.minio.boto3.client")
    @patch("adaptive_router.savers.minio.get_writer")
    def test_save_profile_minio_path(
        self, mock_get_writer, mock_boto3_client, sample_profile
    ):
        """Test saving profile with MinIO path format."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        mock_writer = MagicMock()
        mock_writer.write_to_bytes.return_value = b"test data"
        mock_get_writer.return_value = mock_writer

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        result_path = saver.save_profile(
            sample_profile, "minio://my-bucket/my-profile.json"
        )

        assert result_path == "minio://my-bucket/my-profile.json"
        mock_s3.put_object.assert_called_once_with(
            Bucket="my-bucket",
            Key="my-profile.json",
            Body=b"test data",
            ContentType="application/json",
        )

    @patch("adaptive_router.savers.minio.boto3.client")
    def test_save_profile_invalid_s3_path(self, mock_boto3_client, sample_profile):
        """Test saving profile with invalid S3 path raises ValueError."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        with pytest.raises(ValueError, match="Invalid S3 path format"):
            saver.save_profile(sample_profile, "s3://invalid-path")

        with pytest.raises(ValueError, match="Invalid MinIO path format"):
            saver.save_profile(sample_profile, "minio://invalid-path")

    @patch("adaptive_router.savers.minio.boto3.client")
    def test_save_profile_invalid_path_format(self, mock_boto3_client, sample_profile):
        """Test saving profile with invalid path format raises ValueError."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        with pytest.raises(ValueError, match="Invalid path format"):
            saver.save_profile(sample_profile, "invalid://bucket/key")

    @patch("adaptive_router.savers.minio.boto3.client")
    @patch("adaptive_router.savers.minio.get_writer")
    def test_save_profile_client_error(
        self, mock_get_writer, mock_boto3_client, sample_profile
    ):
        """Test saving profile handles ClientError."""
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket does not exist"}},
            "PutObject",
        )
        mock_boto3_client.return_value = mock_s3

        mock_writer = MagicMock()
        mock_writer.write_to_bytes.return_value = b"test data"
        mock_get_writer.return_value = mock_writer

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        with pytest.raises(IOError, match="MinIO upload failed"):
            saver.save_profile(sample_profile, "s3://my-bucket/my-profile.json")

    @patch("adaptive_router.savers.minio.boto3.client")
    def test_health_check_success(self, mock_boto3_client):
        """Test health check returns True when bucket is accessible."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        result = saver.health_check()
        assert result is True
        mock_s3.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @patch("adaptive_router.savers.minio.boto3.client")
    def test_health_check_failure(self, mock_boto3_client):
        """Test health check returns False when bucket is not accessible."""
        mock_s3 = MagicMock()
        mock_s3.head_bucket.side_effect = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket does not exist"}},
            "HeadBucket",
        )
        mock_boto3_client.return_value = mock_s3

        saver = MinIOProfileSaver(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="test-profile.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        result = saver.health_check()
        assert result is False
        mock_s3.head_bucket.assert_called_once_with(Bucket="test-bucket")
