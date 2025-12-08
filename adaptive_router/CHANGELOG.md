# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-08

### Added
- Initial release of Adaptive Router
- **Core Routing**: Cluster-based intelligent LLM model selection using UniRouter algorithm
- **Feature Extraction**: Semantic embeddings via SentenceTransformers (all-MiniLM-L6-v2)
- **ModelRouter**: Main routing class with `select_model()` and `route()` methods
- **Profile Loaders**: Local file and MinIO/S3 profile loading support
- **Cost Optimization**: Configurable `cost_bias` parameter (0.0=cheapest, 1.0=best quality)
- **Python Library**: Import and use directly in Python applications
- **C++ Core** (optional): High-performance inference via `adaptive-router-core` and `adaptive-router-core-cu12`
- **Training Scripts**: Tools for creating custom router profiles from labeled datasets

### Packages
- `adaptive_router` - Main Python ML library
- `adaptive-router-core` - CPU-only C++ extension (optional, install with `pip install adaptive_router[cpu]`)
- `adaptive-router-core-cu12` - CUDA 12.x C++ extension (optional, install with `pip install adaptive_router[cu12]`)
