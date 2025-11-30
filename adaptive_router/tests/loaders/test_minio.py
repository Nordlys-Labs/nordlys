"""Tests for MinIOProfileLoader."""

import json
from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError

from adaptive_router.loaders.minio import MinIOProfileLoader
from adaptive_router.models.storage import MinIOSettings


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
def minio_settings() -> MinIOSettings:
    """Create MinIO settings for testing."""
    return MinIOSettings(
        bucket_name="test-bucket",
        region="us-east-1",
        profile_key="profiles/test-profile.json",
        endpoint_url="http://localhost:9000",
        root_user="test-user",
        root_password="test-password",
        connect_timeout=5,
        read_timeout=30,
    )


@pytest.fixture
def mock_s3_client():
    """Create a mock S3 client."""
    return MagicMock(spec=boto3.client("s3"))


class TestMinIOProfileLoaderInitialization:
    """Test MinIOProfileLoader initialization."""

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_initialization_with_valid_params(self, mock_boto3_client, mock_s3_client):
        """Test loader initializes with valid parameters."""
        mock_boto3_client.return_value = mock_s3_client

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/test.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        assert loader.bucket_name == "test-bucket"
        assert loader.profile_key == "profiles/test.json"
        assert loader.endpoint_url == "http://localhost:9000"

        mock_boto3_client.assert_called_once()
        call_args = mock_boto3_client.call_args
        assert call_args[1]["endpoint_url"] == "http://localhost:9000"
        assert call_args[1]["aws_access_key_id"] == "test-user"
        assert call_args[1]["aws_secret_access_key"] == "test-password"

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_initialization_with_custom_timeouts(
        self, mock_boto3_client, mock_s3_client
    ):
        """Test loader initializes with custom timeouts."""
        mock_boto3_client.return_value = mock_s3_client

        MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/test.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
            connect_timeout=10,
            read_timeout=60,
        )

        mock_boto3_client.assert_called_once()
        call_args = mock_boto3_client.call_args
        config = call_args[1]["config"]
        assert config.connect_timeout == 10
        assert config.read_timeout == 60

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_from_settings_classmethod(
        self, mock_boto3_client, mock_s3_client, minio_settings
    ):
        """Test from_settings classmethod."""
        mock_boto3_client.return_value = mock_s3_client

        loader = MinIOProfileLoader.from_settings(minio_settings)

        assert loader.bucket_name == "test-bucket"
        assert loader.profile_key == "profiles/test-profile.json"
        assert loader.endpoint_url == "http://localhost:9000"

        mock_boto3_client.assert_called_once()


class TestMinIOProfileLoaderLoadProfile:
    """Test MinIOProfileLoader load_profile method."""

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_load_profile_success(
        self, mock_boto3_client, mock_s3_client, valid_profile_data
    ):
        """Test successful profile loading."""
        mock_boto3_client.return_value = mock_s3_client

        # Mock the S3 response
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps(valid_profile_data).encode("utf-8")
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/test.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        profile = loader.load_profile()

        assert profile.metadata.n_clusters == 2
        assert profile.metadata.silhouette_score == 0.45
        assert len(profile.models) == 1
        assert len(profile.models[0].error_rates) == 2
        assert (
            profile.metadata.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        )

        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket", Key="profiles/test.json"
        )

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_load_profile_with_missing_key(self, mock_boto3_client, mock_s3_client):
        """Test loading profile with missing key raises FileNotFoundError."""
        mock_boto3_client.return_value = mock_s3_client

        # Mock ClientError with NoSuchKey
        nosuchkey_error = ClientError(
            {
                "Error": {
                    "Code": "NoSuchKey",
                    "Message": "The specified key does not exist",
                }
            },
            "GetObject",
        )
        mock_s3_client.get_object.side_effect = nosuchkey_error

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/missing.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        with pytest.raises(FileNotFoundError, match="Profile not found in MinIO"):
            loader.load_profile()

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_load_profile_with_corrupted_json(self, mock_boto3_client, mock_s3_client):
        """Test loading corrupted JSON raises ValueError."""
        mock_boto3_client.return_value = mock_s3_client

        # Mock corrupted JSON response
        mock_body = MagicMock()
        mock_body.read.return_value = b"{invalid json content"
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/corrupted.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        with pytest.raises(ValueError, match="Corrupted JSON"):
            loader.load_profile()

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_load_profile_with_invalid_schema(self, mock_boto3_client, mock_s3_client):
        """Test loading profile with invalid schema raises ValueError."""
        mock_boto3_client.return_value = mock_s3_client

        # Mock invalid schema response
        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"invalid": "schema"}).encode("utf-8")
        mock_s3_client.get_object.return_value = {"Body": mock_body}

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/invalid.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        with pytest.raises(ValueError, match="validation"):
            loader.load_profile()

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_load_profile_with_client_error(self, mock_boto3_client, mock_s3_client):
        """Test loading profile with ClientError raises exception."""
        mock_boto3_client.return_value = mock_s3_client

        # Mock ClientError
        client_error = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}, "GetObject"
        )
        mock_s3_client.get_object.side_effect = client_error

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/test.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        with pytest.raises(ClientError):
            loader.load_profile()


class TestMinIOProfileLoaderHealthCheck:
    """Test MinIOProfileLoader health_check method."""

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_health_check_success(self, mock_boto3_client, mock_s3_client):
        """Test health check returns True when bucket is accessible."""
        mock_boto3_client.return_value = mock_s3_client

        # Mock successful head_bucket
        mock_s3_client.head_bucket.return_value = {}

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/test.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        result = loader.health_check()

        assert result is True
        mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")

    @patch("adaptive_router.loaders.minio.boto3.client")
    def test_health_check_failure(self, mock_boto3_client, mock_s3_client):
        """Test health check returns False when bucket is not accessible."""
        mock_boto3_client.return_value = mock_s3_client

        # Mock ClientError for head_bucket
        client_error = ClientError(
            {"Error": {"Code": "NoSuchBucket", "Message": "Bucket does not exist"}},
            "HeadBucket",
        )
        mock_s3_client.head_bucket.side_effect = client_error

        loader = MinIOProfileLoader(
            bucket_name="test-bucket",
            region="us-east-1",
            profile_key="profiles/test.json",
            endpoint_url="http://localhost:9000",
            access_key_id="test-user",
            secret_access_key="test-password",
        )

        result = loader.health_check()

        assert result is False
        mock_s3_client.head_bucket.assert_called_once_with(Bucket="test-bucket")
