"""Custom exceptions for Adaptive Router Core module.

This module defines a hierarchy of exceptions specific to the Adaptive Router,
providing clearer error messages and better debugging experience for library users.
"""

from __future__ import annotations


class AdaptiveRouterError(Exception):
    """Base exception for all Adaptive Router operations.

    This is the root exception that all other Adaptive Router exceptions inherit from.
    It provides a consistent interface for error handling across the library.
    """


class ModelNotFoundError(AdaptiveRouterError):
    """Raised when a requested model is not available in the router.

    This exception is raised when:
    - A model ID is requested that doesn't exist in the router's model features
    - Model filtering results in no valid models remaining
    - A model is referenced in a profile but not provided in the models list
    """


class ClusterNotFittedError(AdaptiveRouterError):
    """Raised when cluster operations are attempted before fitting.

    This exception is raised when:
    - predict() or assign_single() is called before fit()
    - Cluster engine operations require fitted state
    """


class ClusterNotConfiguredError(AdaptiveRouterError):
    """Raised when ClusterEngine methods are called before configuration.

    This exception is raised when:
    - fit(), predict(), or assign_single() called without configure()
    - Components (SentenceTransformer, KMeans) not initialized
    - Configuration parameters not set before training
    """


class ProfileLoadError(AdaptiveRouterError):
    """Raised when profile loading or parsing fails.

    This exception is raised when:
    - MinIO profile loading fails
    - Local file profile loading fails
    - Profile data is corrupted or invalid
    - Required profile components are missing
    """


class InvalidModelFormatError(AdaptiveRouterError):
    """Raised when model ID format is invalid.

    This exception is raised when:
    - Model ID doesn't follow 'provider/model_name' format
    - Provider or model name is empty
    - Model ID parsing fails during routing
    """


class FeatureExtractionError(AdaptiveRouterError):
    """Raised when feature extraction fails.

    This exception is raised when:
    - Embedding model loading fails
    - TF-IDF vectorization fails
    - Feature normalization fails
    - Input validation fails for feature extraction
    """
