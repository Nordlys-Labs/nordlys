# Nordlys Core

High-performance C++ library for intelligent LLM model routing and selection.

## Overview

Nordlys Core provides:
- **Fast routing**: Sub-millisecond model selection with GPU acceleration
- **Cross-platform**: Linux, macOS, and Windows
- **Language bindings**: Python and C FFI APIs
- **Smart selection**: K-means clustering with cost-accuracy optimization

## Build Requirements

- **Compiler**: C++20 compatible (GCC 10+, Clang 12+, MSVC 19.3+)
- **CMake**: 3.24 or higher
- **Conan**: 2.0 or higher
- **CUDA**: Optional, version 12.x for GPU acceleration

## Quick Start

```bash
# Install Conan
pip install conan

# Install dependencies
conan install . --build=missing -s compiler.cppstd=20

# Configure and build
cmake --preset conan-release -DNORDLYS_BUILD_C=ON
cmake --build --preset conan-release
```

**Requirements:**
- C++20 compiler (GCC 10+, Clang 12+, MSVC 19.3+)
- CMake 3.24+
- Conan 2.0+
- CUDA 12.x (optional, Linux only)

### Build Outputs

- **Module Libraries** - Per-domain static libraries (see Architecture below)
- **C FFI** (`libnordlys_c.so`/`.dylib`) - C-compatible API for other languages
- **Python Extension** (`nordlys_core_ext.so`) - Python bindings via nanobind

## Build Options

- `DNORDLYS_BUILD_PYTHON=ON|OFF` - Python bindings (default: ON)
- `DNORDLYS_BUILD_C=ON|OFF` - C FFI bindings (default: OFF)
- `DNORDLYS_BUILD_TESTS=ON|OFF` - Test suite (default: OFF)
- `DNORDLYS_BUILD_BENCHMARKS=ON|OFF` - Benchmark suite (default: OFF)
- `DNORDLYS_ENABLE_CUDA=ON|OFF` - CUDA support (default: OFF, Linux only)

## Testing

```bash
# Run all tests
cmake --build . --target test

# Run specific test suites
ctest --output-on-failure

# Run tests with GPU (if CUDA enabled)
ctest -R cuda --output-on-failure
```

## Benchmarking

Run performance benchmarks to measure routing latency and throughput:

```bash
# Build with benchmarks enabled
cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON
cmake --build --preset conan-release

# Run all benchmarks
./build/Release/benchmarks/bench_nordlys_core

# Run specific benchmark pattern
./build/Release/benchmarks/bench_nordlys_core --benchmark_filter=RoutingSingle

# Save results to JSON
./build/Release/benchmarks/bench_nordlys_core \
  --benchmark_format=json \
  --benchmark_out=results.json
```

Benchmarks measure:
- **Single routing latency** across different profile sizes
- **Batch routing throughput** for high-load scenarios
- **Checkpoint loading** and router initialization time
- **Concurrent routing** performance with multiple threads

See [benchmarks/README.md](benchmarks/README.md) for detailed documentation.

## Architecture

Nordlys Core is organized into domain-specific modules:

```
nordlys-core/
├── common/        # Shared types: Matrix, Device, Result
├── scoring/       # Model scoring with cost-accuracy optimization
├── checkpoint/    # Checkpoint serialization (JSON/MessagePack)
├── clustering/    # K-means clustering with CPU/CUDA backends
├── routing/       # High-level routing API (Nordlys class)
├── bindings/      # Language bindings (Python, C FFI)
├── benchmarks/    # Performance benchmarks
└── test/          # Integration tests
```

### Module Dependencies

```
routing → (clustering, scoring, checkpoint)
checkpoint → (common, scoring)
clustering → common
scoring → common
common → (no dependencies)
```

### CMake Targets

| Target | Type | Description |
|--------|------|-------------|
| `Nordlys::Common` | INTERFACE | Header-only shared types |
| `Nordlys::Scoring` | STATIC | Model scoring library |
| `Nordlys::Checkpoint` | STATIC | Checkpoint I/O library |
| `Nordlys::Clustering` | STATIC | Clustering with CUDA support |
| `Nordlys::Routing` | STATIC | High-level routing API |
| `Nordlys::Core` | INTERFACE | Unified interface (links all modules) |

### Include Paths

```cpp
#include <nordlys/common/matrix.hpp>
#include <nordlys/scoring/scorer.hpp>
#include <nordlys/checkpoint/checkpoint.hpp>
#include <nordlys/clustering/cluster.hpp>
#include <nordlys/routing/nordlys.hpp>
```

## Module Documentation

- [Common](common/README.md) - Shared types and utilities
- [Scoring](scoring/README.md) - Model scoring algorithm
- [Checkpoint](checkpoint/README.md) - Checkpoint serialization
- [Clustering](clustering/README.md) - K-means clustering engine
- [Routing](routing/README.md) - High-level routing API
- [Bindings](bindings/README.md) - Language bindings overview
  - [Python](bindings/python/README.md) - Python API
  - [C FFI](bindings/c/README.md) - C API for other languages
- [Benchmarks](benchmarks/README.md) - Performance benchmarks

## Profiling

Profile benchmarks using headless profiling tools (`perf` or `gprof`) that work without GUI connections:

```bash
# Build benchmarks
cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON
cmake --build --preset conan-release

# Profile with perf (recommended)
perf record ./build/Release/benchmarks/bench_nordlys_core --benchmark_filter=RoutingSingle_Medium
perf report
```

**Profiling tools provide:**
- Function-level timing analysis
- Call graph visualization
- Hotspot identification
- Works in CI/headless environments
- No GUI connection required

See [benchmarks/PROFILING.md](benchmarks/PROFILING.md) for detailed guide.

## Usage

### C++ API

```cpp
#include <nordlys/routing/nordlys.hpp>

auto checkpoint = NordlysCheckpoint::from_json_file("checkpoint.json");
auto router = Nordlys::from_checkpoint(std::move(checkpoint));
auto result = router.route(embedding, embedding_size, 0.5f);
```

### Python API

```python
from nordlys_core import Nordlys, NordlysCheckpoint

checkpoint = NordlysCheckpoint.from_json_file("checkpoint.json")
router = Nordlys.from_checkpoint(checkpoint)
result = router.route(embedding, cost_bias=0.5)
```

### C FFI API

```c
#include "nordlys.h"

NordlysRouter* router = nordlys_router_create_from_file("checkpoint.json", ...);
NordlysRouteResult* result = nordlys_router_route(router, embedding, ...);
```

See module READMEs for detailed API documentation:
- [Routing](routing/README.md) - C++ API
- [Python Bindings](bindings/python/README.md) - Python API
- [C FFI](bindings/c/README.md) - C API

## Performance

- **Routing latency**: ~50-500us (depends on profile size)
- **Memory**: ~10-50MB (depends on profile size)
- **GPU acceleration**: 10-100x speedup for batch operations
- **Thread-safe**: Safe for concurrent routing

## Dependencies

- **nlohmann/json** - JSON parsing
- **msgpack-cxx** - MessagePack serialization
- **simdjson** - Fast JSON parsing
- **USearch** - Vector search (CPU backend)
- **SimSIMD** - SIMD-accelerated distance computations
- **nanobind** - Python bindings (optional)
- **CUDA Toolkit** - GPU support (optional)

## Contributing

See the main [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Links

- **Documentation**: https://docs.nordlyslabs.com
- **Issues**: https://github.com/Nordlys-Labs/nordlys/issues
- **Main Repository**: https://github.com/Nordlys-Labs/nordlys
