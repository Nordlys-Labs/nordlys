"""Basic tests for Router class initialization."""

import pytest

from nordlys import Router


class TestNordlysInitialization:
    """Test Router class initialization."""

    def test_create_with_valid_models(self, sample_models):
        """Test creating Router with valid model configurations."""
        nordlys = Router(models=sample_models)
        assert nordlys is not None

    def test_create_with_empty_models_fails(self):
        """Test that creating Router with empty models list fails."""
        with pytest.raises(ValueError, match="At least one model"):
            Router(models=[])

    def test_create_with_custom_nr_clusters(self, sample_models):
        """Test creating Router with custom number of clusters."""
        nordlys = Router(models=sample_models, nr_clusters=15)
        assert nordlys is not None

    def test_create_with_custom_random_state(self, sample_models):
        """Test creating Router with custom random state."""
        nordlys = Router(models=sample_models, random_state=123)
        assert nordlys is not None


class TestNordlysAttributes:
    """Test Router instance attributes."""

    def test_nordlys_stores_models(self, sample_models):
        """Test that Router stores model configurations."""
        nordlys = Router(models=sample_models)
        # Access private attribute for testing
        assert len(nordlys._models) == len(sample_models)
        assert len(nordlys._model_ids) == len(sample_models)

    def test_nordlys_default_embedding_model(self, sample_models):
        """Test that Router has default embedding model."""
        nordlys = Router(models=sample_models)
        # Check default embedding model name
        assert nordlys._embedding_model_name is not None
        assert "MiniLM" in nordlys._embedding_model_name

    def test_nordlys_default_nr_clusters(self, sample_models):
        """Test default number of clusters."""
        nordlys = Router(models=sample_models)
        assert nordlys._nr_clusters == 20  # Default value

    def test_nordlys_custom_embedding_model(self, sample_models):
        """Test Router with custom embedding model name."""
        nordlys = Router(
            models=sample_models,
            embedding_model="sentence-transformers/paraphrase-MiniLM-L3-v2",
        )
        assert "paraphrase" in nordlys._embedding_model_name

    def test_nordlys_embedding_model_loaded_at_init(self, sample_models):
        """Test that embedding model is loaded at initialization."""
        nordlys = Router(models=sample_models)
        # Embedding model should be loaded (not None)
        assert nordlys._embedding_model is not None
        # Should be a SentenceTransformer instance
        from sentence_transformers import SentenceTransformer

        assert isinstance(nordlys._embedding_model, SentenceTransformer)


class TestNordlysState:
    """Test Router fitted state."""

    def test_nordlys_not_fitted_initially(self, sample_models):
        """Test that Router is not fitted initially."""
        nordlys = Router(models=sample_models)
        assert nordlys._is_fitted is False

    def test_nordlys_embeddings_none_initially(self, sample_models):
        """Test that embeddings are None before fitting."""
        nordlys = Router(models=sample_models)
        assert nordlys._embeddings is None
        assert nordlys._reduced_embeddings is None
        assert nordlys._labels is None
