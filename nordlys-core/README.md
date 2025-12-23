# Nordlys Model Engine - C++ Core

High-performance C++ core library for intelligent LLM model routing and selection. This is the core engine used by Nordlys model runtime and tooling.

## Overview

The Nordlys C++ core provides:
- **High-performance inference**: Fast model routing with GPU acceleration
- **Cross-platform support**: Linux, macOS, and Windows
- **Language bindings**: Python (via nanobind) and C FFI APIs
- **ML routing**: UniRouter algorithm with K-means clustering
- **Cost optimization**: Intelligent balancing of model capability vs cost

## Build Requirements

- **Compiler**: C++20 compatible (GCC 10+, Clang 12+, MSVC 19.3+)
- **CMake**: 3.24 or higher
- **Conan**: 2.0 or higher
- **CUDA**: Optional, version 12.x for GPU acceleration

## Quick Start

### Prerequisites
```bash
# Install Conan (package manager for C++)
pip install conan

# For CUDA support (Linux only)
# Install CUDA Toolkit 12.x from NVIDIA
```

### Building the Core Library

```bash
cd nordlys-core

# Install dependencies with Conan
conan install . --build=missing -s compiler.cppstd=20

# Configure with CMake
cmake --preset conan-release -DNORDLYS_BUILD_C=ON

# Build the libraries
cmake --build . --target nordlys_core nordlys_c
```

### Build Targets

The build process creates several artifacts:

1. **Core Library** (`libnordlys_core.a`)
   - Static C++ library with routing logic
   - Used by Python bindings and C FFI

2. **C FFI Library** (`libnordlys_c.so` or `nordlys_c.dll`)
   - C-compatible API for integration with other languages
   - Ideal for Rust, Go, Java, and other systems

3. **Python Extension** (`nordlys_core_ext.so`)
   - Python bindings via nanobind
   - Used by the `nordlys` Python package

## Building Variants

### CPU-only Version (Default)
```bash
cmake --preset conan-release
cmake --build .
```

### CUDA Version (Linux only)
```bash
cmake --preset conan-release -DNORDLYS_ENABLE_CUDA=ON
cmake --build .
```

### Package Options

- `DNORDLYS_BUILD_PYTHON=ON|OFF` - Build Python bindings (default: ON)
- `DNORDLYS_BUILD_C=ON|OFF` - Build C FFI bindings (default: OFF)
- `DNORDLYS_BUILD_TESTS=ON|OFF` - Build test suite (default: OFF)
- `DNORDLYS_ENABLE_CUDA=ON|OFF` - Enable CUDA support (default: OFF)

## Testing

```bash
# Run all tests
cmake --build . --target test

# Run specific test suites
ctest --output-on-failure

# Run tests with GPU (if CUDA enabled)
ctest -R cuda --output-on-failure
```

## Usage Examples

### C API Usage

```c
#include "nordlys.h"

// Create router from profile
NordlysRouter* router = nordlys_router_create("profile.json");

// Route with embedding
float embedding[] = {0.1f, 0.2f, ...};
NordlysErrorCode error;
NordlysRouteResult32* result = nordlys_router_route_f32(
    router, embedding, embedding_size, 0.5f, &error);

// Use result
printf("Selected model: %s\n", result->selected_model);

// Cleanup
nordlys_route_result_free_f32(result);
nordlys_router_destroy(router);
```

### Integration with Python

```python
# The C++ core is automatically used by the Python package
from nordlys import ModelRouter

router = ModelRouter.from_file("profile.json")
response = router.select_model(prompt="Write a Python function")
print(f"Selected: {response.model_id}")
```

## Architecture

The C++ core implements the following algorithms:

1. **UniRouter Algorithm**: Cluster-based model selection
2. **K-means Clustering**: Prompt similarity grouping
3. **Feature Extraction**: Embedding computation
4. **Cost Optimization**: λ-weighted scoring

## Performance Characteristics

- **Routing latency**: ~0.15ms (excluding embedding computation)
- **Memory usage**: ~10-50MB depending on profile size
- **GPU acceleration**: 10-100x speedup for batch operations
- **Concurrency**: Thread-safe for concurrent routing operations

## Dependencies

The C++ core depends on:

- **Eigen3**: Linear algebra
- **Boost**: Utilities and containers
- **nlohmann/json**: JSON parsing
- **msgpack-cxx**: Serialization
- **tsl-robin-map**: High-performance hash maps
- **nanobind**: Python bindings (optional)
- **CUDA Toolkit**: GPU support (optional)

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Links

- **Documentation**: https://docs.llmadaptive.uk
- **Issues**: https://github.com/Nordlys-Labs/nordlys/issues
- **Main Repository**: https://github.com/Nordlys-Labs/nordlys
