"""Tests for package initialization and imports."""

import nordlys


class TestPackageImports:
    """Test that all public API imports work correctly."""

    def test_import_nordlys(self):
        """Test importing the main Router class."""
        from nordlys import Router

        assert Router is not None

    def test_import_route_result(self):
        """Test importing RouteResult."""
        from nordlys import RouteResult

        assert RouteResult is not None

    def test_import_reduction_module(self):
        """Test importing reduction module."""
        from nordlys import reduction

        assert reduction is not None

    def test_import_clustering_module(self):
        """Test importing clustering module."""
        from nordlys import clustering

        assert clustering is not None

    def test_import_embeddings_module(self):
        """Test importing embeddings module."""
        from nordlys import embeddings

        assert embeddings is not None


class TestPackageMetadata:
    """Test package metadata."""

    def test_version_exists(self):
        """Test that __version__ is defined."""
        version = nordlys.__version__
        assert isinstance(version, str)

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        version = nordlys.__version__
        parts = version.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        expected_exports = {
            "Router",
            "Dataset",
            "Trainer",
            "RouteResult",
            "NordlysCheckpoint",
            "TrainingMetrics",
            "EmbeddingConfig",
            "ClusteringConfig",
            "ModelFeatures",
            "reduction",
            "clustering",
            "embeddings",
        }
        exports = nordlys.__all__
        assert set(exports) == expected_exports
