"""Tests for package initialization and imports."""

import nordlys


class TestPackageImports:
    """Test that all public API imports work correctly."""

    def test_import_nordlys(self):
        """Test importing the main Nordlys class."""
        from nordlys import Nordlys

        assert Nordlys is not None

    def test_import_model_config(self):
        """Test importing ModelConfig."""
        from nordlys import ModelConfig

        assert ModelConfig is not None

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


class TestPackageMetadata:
    """Test package metadata."""

    def test_version_exists(self):
        """Test that __version__ is defined."""
        assert hasattr(nordlys, "__version__")
        assert isinstance(nordlys.__version__, str)

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        version = nordlys.__version__
        parts = version.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        expected_exports = {
            "Nordlys",
            "ModelConfig",
            "RouteResult",
            "NordlysCheckpoint",
            "TrainingMetrics",
            "EmbeddingConfig",
            "ClusteringConfig",
            "ModelFeatures",
            "reduction",
            "clustering",
        }
        assert hasattr(nordlys, "__all__")
        assert set(nordlys.__all__) == expected_exports
