"""
Profile loading implementations for Adaptive Router.

This module provides loaders for reading router profiles from various storage backends.
Profiles contain cluster centers, model configurations, and per-cluster error rates.

Classes
-------
ProfileLoader
    Abstract base class for custom profile loader implementations.
    Subclass this to implement custom storage backends.

LocalFileProfileLoader
    Load profiles from local JSON files.
    Suitable for development, testing, and single-server deployments.

MinIOProfileLoader
    Load profiles from MinIO/S3-compatible storage with connection pooling.
    Suitable for production deployments with distributed systems.

Example
-------
Loading from local file:

    >>> from adaptive_router.loaders import LocalFileProfileLoader
    >>>
    >>> loader = LocalFileProfileLoader("profile.json")
    >>> profile = loader.load_profile()
    >>> print(f"Loaded profile with {profile.metadata.n_clusters} clusters")
    Loaded profile with 20 clusters

Loading from MinIO/S3:

    >>> from adaptive_router.loaders import MinIOProfileLoader
    >>> from adaptive_router.models import MinIOSettings
    >>>
    >>> settings = MinIOSettings(
    ...     endpoint_url="https://minio.example.com",
    ...     root_user="admin",
    ...     root_password="password",
    ...     bucket_name="profiles",
    ...     object_key="router_profile.json"
    ... )
    >>> loader = MinIOProfileLoader(settings)
    >>> profile = loader.load_profile()

Custom loader implementation:

    >>> from adaptive_router.loaders import ProfileLoader
    >>> from adaptive_router.models.storage import RouterProfile
    >>>
    >>> class DatabaseProfileLoader(ProfileLoader):
    ...     def load_profile(self) -> RouterProfile:
    ...         # Load from database
    ...         data = fetch_from_database()
    ...         return RouterProfile(**data)

See Also
--------
adaptive_router.core.router : ModelRouter class that uses these loaders
adaptive_router.models.storage : RouterProfile data model
adaptive_router.savers : Profile saving implementations
"""
from adaptive_router.loaders.base import ProfileLoader
from adaptive_router.loaders.local import LocalFileProfileLoader
from adaptive_router.loaders.minio import MinIOProfileLoader

__all__ = ["ProfileLoader", "LocalFileProfileLoader", "MinIOProfileLoader"]
