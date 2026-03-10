"""Basic tests for Router class initialization."""

import pytest

from nordlys import Router


class TestRouterInitialization:
    """Test Router class initialization."""

    def test_create_with_valid_models(self, sample_models):
        """Test creating Router with valid model configurations."""
        nordlys = Router(models=sample_models)
        assert nordlys is not None

    def test_create_with_empty_models_fails(self):
        """Test that creating Router with empty models list fails."""
        with pytest.raises(ValueError, match="At least one model"):
            Router(models=[])


class TestRouterAttributes:
    """Test Router instance attributes."""

    def test_router_stores_models(self, sample_models):
        """Test that Router stores model configurations."""
        nordlys = Router(models=sample_models)
        # Access private attribute for testing
        assert len(nordlys._models) == len(sample_models)
        assert len(nordlys._model_ids) == len(sample_models)

    def test_router_default_embedding_model(self, sample_models):
        """Test that Router has default embedding model."""
        nordlys = Router(models=sample_models)
        # Check default embedding model name
        assert nordlys._embedding_model_name is not None
        assert "MiniLM" in nordlys._embedding_model_name

    def test_router_default_runtime_metadata(self, sample_models):
        """Test runtime metadata defaults before loading checkpoint."""
        nordlys = Router(models=sample_models)
        assert nordlys._nr_clusters == 0
        assert nordlys._random_state == 0

    def test_router_custom_embedding_model(self, sample_models):
        """Test Router with custom embedding model name."""
        nordlys = Router(
            models=sample_models,
            embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        )
        assert "paraphrase" in nordlys._embedding_model_name

    def test_router_embedding_model_loaded_at_init(self, sample_models):
        """Test that embedding model is loaded at initialization."""
        nordlys = Router(models=sample_models)
        # Embedding model should be loaded (not None)
        assert nordlys._embedding_model is not None
        # Should be a SentenceTransformer instance
        from sentence_transformers import SentenceTransformer

        assert isinstance(nordlys._embedding_model, SentenceTransformer)


class TestRouterState:
    """Test Router fitted state."""

    def test_router_not_fitted_initially(self, sample_models):
        """Test that Router is not fitted initially."""
        nordlys = Router(models=sample_models)
        assert nordlys._is_fitted is False

    def test_router_has_no_training_methods(self, sample_models):
        """Test that Router exposes runtime-only API."""
        nordlys = Router(models=sample_models)
        with pytest.raises(AttributeError):
            nordlys.fit  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            nordlys.fit_transform  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            nordlys.transform  # type: ignore[attr-defined]
