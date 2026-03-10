"""Basic tests for Router class initialization."""

import pytest

from nordlys import Router


class TestRouterInitialization:
    """Test Router class initialization."""

    def test_create_with_checkpoint(self, sample_checkpoint):
        """Test creating Router from checkpoint."""
        nordlys = Router(checkpoint=sample_checkpoint)
        assert nordlys is not None

    def test_create_with_invalid_checkpoint_path_fails(self):
        """Test that creating Router with bad checkpoint path fails."""
        with pytest.raises(RuntimeError, match="Failed to open checkpoint file"):
            Router(checkpoint="missing-checkpoint.json")


class TestRouterAttributes:
    """Test Router instance attributes."""

    def test_router_stores_models(self, sample_models, sample_checkpoint):
        """Test that Router stores model configurations."""
        nordlys = Router(checkpoint=sample_checkpoint)
        # Access private attribute for testing
        assert len(nordlys._models) == len(sample_models)
        assert len(nordlys._model_ids) == len(sample_models)

    def test_router_embedding_model_from_checkpoint(self, sample_checkpoint):
        """Test that Router has default embedding model."""
        nordlys = Router(checkpoint=sample_checkpoint)
        # Check default embedding model name
        assert nordlys._embedding_model_name is not None
        assert "MiniLM" in nordlys._embedding_model_name

    def test_router_runtime_metadata_from_checkpoint(self, sample_checkpoint):
        """Test runtime metadata is loaded from checkpoint."""
        nordlys = Router(checkpoint=sample_checkpoint)
        assert nordlys._nr_clusters == 2
        assert nordlys._random_state == 42

    def test_router_embedding_model_loaded_at_init(self, sample_checkpoint):
        """Test that embedding model is loaded at initialization."""
        nordlys = Router(checkpoint=sample_checkpoint)
        # Embedding model should be loaded (not None)
        assert nordlys._embedding_model is not None
        # Should be a SentenceTransformer instance
        from sentence_transformers import SentenceTransformer

        assert isinstance(nordlys._embedding_model, SentenceTransformer)


class TestRouterState:
    """Test Router fitted state."""

    def test_router_is_ready_after_constructor_load(self, sample_checkpoint):
        """Test that Router is initialized after constructor checkpoint load."""
        nordlys = Router(checkpoint=sample_checkpoint)
        assert nordlys._core_engine is not None

    def test_router_has_no_training_methods(self, sample_checkpoint):
        """Test that Router exposes runtime-only API."""
        nordlys = Router(checkpoint=sample_checkpoint)
        with pytest.raises(AttributeError):
            nordlys.fit  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            nordlys.fit_transform  # type: ignore[attr-defined]
        with pytest.raises(AttributeError):
            nordlys.transform  # type: ignore[attr-defined]
