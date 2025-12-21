# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2025-12-17

### Changed
- **PyTorch Optional Dependencies**: Updated to support CPU and CUDA variants using uv index-based distribution
  - CPU extra: `torch>=2.5.1` from pytorch-cpu index
  - CUDA extra: `torch>=2.5.1` from pytorch-cu128 index with full NVIDIA CUDA 12.8 stack
  - Added uv conflicts to prevent installing both extras simultaneously

## [0.1.4] - 2025-12-17

### Changed
- **Scripts Organization**: Moved scripts directory to same level as train and tests
- **Core Version**: Updated adaptive-router-core dependency to >=0.1.10

### Fixed
- **Remote Code Execution**: Fixed trust_remote_code parameter not being properly handled in msgpack profiles

## [0.1.3] - 2025-12-17

### Changed
- **API Refactor**: Simplified ModelRouter loading API
  - Added `ModelRouter.from_file()` - unified method for loading JSON/MessagePack profiles
  - Added `ModelRouter.profile` property - access loaded RouterProfile directly
  - Removed `from_json_file()` and `from_msgpack_file()` methods
  - Auto-detect file format based on extension (.json/.msgpack)

### Improved
- **Developer Experience**: Cleaner API with single loading method and direct profile access
- **Performance**: Eliminated redundant profile parsing in applications

## [0.1.0] - 2025-12-08

### Added
- Initial release of Adaptive Router
- **Core Routing**: Cluster-based intelligent LLM model selection using UniRouter algorithm
- **Feature Extraction**: Semantic embeddings via SentenceTransformers (all-MiniLM-L6-v2)
- **ModelRouter**: Main routing class with `select_model()` and `route()` methods
- **Profile Loading**: Load profiles from JSON or MessagePack files via factory methods
- **Cost Optimization**: Configurable `cost_bias` parameter (0.0=cheapest, 1.0=best quality)
- **Python Library**: Import and use directly in Python applications
- **C++ Core** (optional): High-performance inference via `adaptive-router-core` and `adaptive-router-core-cu12`
- **Training Scripts**: Tools for creating custom router profiles from labeled datasets

### Packages
- `adaptive_router` - Main Python ML library
- `adaptive-router-core` - CPU-only C++ extension (optional, install with `pip install adaptive_router[cpu]`)
- `adaptive-router-core-cu12` - CUDA 12.x C++ extension (optional, install with `pip install adaptive_router[cu12]`)
