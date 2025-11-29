"""Tests for FeatureExtractor."""

import numpy as np
import pytest

from adaptive_router.core import FeatureExtractor
from adaptive_router.exceptions.core import FeatureExtractionError


@pytest.fixture
def sample_questions() -> list[str]:
    """Create sample code questions for testing."""
    return [
        "How do I sort a list in Python?",
        "What is a lambda function?",
        "How to reverse a string in JavaScript?",
        "Explain async/await in Python",
        "How to use map in JavaScript?",
    ]


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    """Create a FeatureExtractor with test configuration."""
    return FeatureExtractor(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    )


class TestFeatureExtractorInitialization:
    """Test FeatureExtractor initialization."""

    def test_default_initialization(self) -> None:
        """Test FeatureExtractor initializes with default parameters."""
        extractor = FeatureExtractor()
        assert (
            extractor.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert not extractor.is_fitted

    def test_custom_parameters(self) -> None:
        """Test FeatureExtractor with custom parameters."""
        extractor = FeatureExtractor(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=64,
        )
        assert extractor.batch_size == 64
        assert (
            extractor.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        )


class TestFeatureExtractorFit:
    """Test FeatureExtractor fitting."""

    def test_fit_updates_state(
        self, feature_extractor: FeatureExtractor, sample_questions: list[str]
    ) -> None:
        """Test that fit updates extractor state correctly."""
        assert not feature_extractor.is_fitted

        feature_extractor.fit_transform(sample_questions)

        assert feature_extractor.is_fitted

    def test_fit_returns_self(
        self, feature_extractor: FeatureExtractor, sample_questions: list[str]
    ) -> None:
        """Test that fit returns self for method chaining."""
        result = feature_extractor.fit_transform(sample_questions)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == len(sample_questions)

    def test_fit_with_empty_questions(
        self, feature_extractor: FeatureExtractor
    ) -> None:
        """Test fit handles empty questions list."""
        with pytest.raises(FeatureExtractionError):
            feature_extractor.fit_transform([])


class TestFeatureExtractorTransform:
    """Test FeatureExtractor transformation."""

    def test_transform_single_question(
        self, feature_extractor: FeatureExtractor, sample_questions: list[str]
    ) -> None:
        """Test transforming a single question."""
        feature_extractor.fit_transform(sample_questions)

        new_question = "How to use list comprehension in Python?"
        features = feature_extractor.transform([new_question])

        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == 1
        assert features.shape[1] > 0

    def test_transform_before_fit_raises_error(
        self, feature_extractor: FeatureExtractor
    ) -> None:
        """Test transform raises error when not fitted."""
        question = "Test question"

        with pytest.raises(
            FeatureExtractionError,
            match="Must call fit_transform\\(\\) before transform",
        ):
            feature_extractor.transform([question])


class TestFeatureExtractorFitTransform:
    """Test FeatureExtractor fit_transform."""

    def test_fit_transform(
        self, feature_extractor: FeatureExtractor, sample_questions: list[str]
    ) -> None:
        """Test fit_transform combines fit and transform."""
        features = feature_extractor.fit_transform(sample_questions)

        assert feature_extractor.is_fitted
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == len(sample_questions)

    def test_fit_transform_equals_fit_then_transform(
        self, sample_questions: list[str]
    ) -> None:
        """Test fit_transform produces same result as fit then transform."""
        extractor1 = FeatureExtractor()
        extractor2 = FeatureExtractor()

        features1 = extractor1.fit_transform(sample_questions)

        extractor2.fit_transform(sample_questions)
        features2 = extractor2.transform(sample_questions)

        np.testing.assert_array_almost_equal(features1, features2)


class TestFeatureExtractorEdgeCases:
    """Test FeatureExtractor edge cases."""

    def test_transform_with_special_characters(
        self, feature_extractor: FeatureExtractor, sample_questions: list[str]
    ) -> None:
        """Test transformation handles special characters."""
        feature_extractor.fit_transform(sample_questions)

        special_question = "How to use @decorator and #comment in Python?"
        features = feature_extractor.transform([special_question])

        assert isinstance(features, np.ndarray)
        assert not np.isnan(features).any()

    def test_transform_with_empty_question_raises_error(
        self, feature_extractor: FeatureExtractor, sample_questions: list[str]
    ) -> None:
        """Test transformation raises error for empty question text."""
        feature_extractor.fit_transform(sample_questions)

        empty_question = ""
        with pytest.raises(FeatureExtractionError, match="Empty text at index"):
            feature_extractor.transform([empty_question])
