# Adaptive Router Core - C++ Inference Engine

High-performance C++ inference core for Adaptive Router with Python bindings and C FFI API.

## Overview

The Adaptive Router Core is a C++23 implementation of the cluster-based routing algorithm, providing 10x performance improvements over pure Python implementations. It features:

- **Ultra-fast cluster assignment** using Eigen for linear algebra
- **Zero-copy Python integration** via nanobind
- **C FFI API** for integration with other languages
- **Zero heap allocations** per routing request
- **Cross-platform** support (Linux, macOS, Windows)

## Architecture

### Components

1. **Router** (`router.hpp/cpp`) - Main routing API with factory methods and route execution
2. **ClusterEngine** (`cluster.hpp/cpp`) - K-means cluster assignment using Euclidean distance
3. **ModelScorer** (`scorer.hpp/cpp`) - Cost-accuracy optimization with configurable trade-offs
4. **Profile** (`profile.hpp/cpp`) - Profile loading from JSON and MessagePack formats

### Data Flow

```
Embedding (384D) → ClusterEngine → Cluster ID → ModelScorer → Ranked Models → Selected Model
```

## Requirements

### Build Requirements

- **C++23 compiler**:
  - GCC 12+ (Linux)
  - Clang 15+ (macOS/Linux)
  - MSVC 19.30+ / VS 2022 17.0+ (Windows)
- **CMake** 3.24+
- **Conan** 2.x (package manager)

### Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Eigen | 5.0.0 | Linear algebra (matrix operations) |
| nlohmann_json | 3.12.0 | JSON parsing |
| msgpack-cxx | 7.0.0 | Binary serialization |
| nanobind | 2.9.2+ | Python bindings |
| gtest | 1.17.0 | Testing framework |

## Installation

### Building from Source

**Linux/macOS:**

```bash
cd adaptive_router_core

# Install dependencies via Conan
conan install . --output-folder=build --build=missing --settings=build_type=Release

# Configure CMake (use conan-release preset)
cmake --preset conan-release

# Build
cmake --build build/build/Release

# Run tests (optional)
ctest --test-dir build/build/Release --output-on-failure
```

**Windows:**

```bash
cd adaptive_router_core

# Install dependencies via Conan
conan install . --output-folder=build --build=missing --settings=build_type=Release

# Configure CMake (use conan-default preset on Windows)
cmake --preset conan-default

# Build
cmake --build build

# Run tests (optional)
ctest --test-dir build --output-on-failure
```

**Note**: Conan generates different preset names on Windows (`conan-default`) vs Linux/macOS (`conan-release`, `conan-debug`).

### Python Package Installation

```bash
# Install from source
pip install ./adaptive_router_core

# Or using uv
uv pip install ./adaptive_router_core
```

The Python package (`adaptive-router-core`) provides the `adaptive_core_ext` module with Python bindings.

## Usage

### C++ Usage

```cpp
#include "router.hpp"

int main() {
  // Load router from JSON profile
  auto router = Router::from_file("profile.json");

  // Pre-computed embedding (384D from all-MiniLM-L6-v2)
  std::vector<float> embedding = {...};

  // Route with cost_bias (0.0 = prefer accuracy, 1.0 = prefer low cost)
  RouteRequest request{
    .embedding = embedding,
    .cost_bias = 0.5f
  };
  auto response = router.route(request);

  std::cout << "Selected: " << response.selected_model << std::endl;
  std::cout << "Cluster: " << response.cluster_id << std::endl;
  std::cout << "Distance: " << response.cluster_distance << std::endl;

  // Print alternatives
  for (const auto& alt : response.alternatives) {
    std::cout << "Alternative: " << alt << std::endl;
  }

  return 0;
}
```

### Python Usage (nanobind)

```python
from adaptive_core_ext import Router
from sentence_transformers import SentenceTransformer
import numpy as np

# Load sentence transformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load router from profile
router = Router.from_file("profile.json")

# Compute embedding and route
prompt = "Explain quantum computing"
embedding = model.encode(prompt)  # Returns numpy array

# Route with cost bias (zero-copy for float32 C-contiguous arrays)
response = router.route(embedding, cost_bias=0.5)

print(f"Selected: {response.selected_model}")
print(f"Cluster: {response.cluster_id}")
print(f"Distance: {response.cluster_distance}")
print(f"Alternatives: {response.alternatives}")
```

**NumPy Integration:**

```python
import numpy as np
from adaptive_core_ext import Router

router = Router.from_file("profile.json")

# Accepts any floating point numpy array (float32, float64, etc.)
# Automatically converts to float32 C-contiguous if needed
embedding = np.random.randn(384)  # float64 by default
response = router.route(embedding, 0.5)

# Or use float32 directly (avoids conversion overhead)
embedding = np.random.randn(384).astype(np.float32)
response = router.route(embedding, 0.5)

# Batch routing: process multiple embeddings efficiently
embeddings = np.random.randn(100, 384)
responses = router.route_batch(embeddings, cost_bias=0.5)

print(f"Processed {len(responses)} embeddings")
for i, resp in enumerate(responses):
    print(f"[{i}] Selected: {resp.selected_model}")
```

**Performance Note:** For best performance, use float32 C-contiguous arrays to avoid conversion overhead.

### C API (FFI)

For integration with other languages (Rust, Go, Julia, etc.):

```c
#include "adaptive.h"
#include <stdio.h>

int main() {
  // Create router from profile
  AdaptiveRouter* router = adaptive_router_create("profile.json");
  if (!router) {
    fprintf(stderr, "Failed to load profile\n");
    return 1;
  }

  // Pre-computed embedding
  float embedding[384] = {...};

  // Route with cost_bias
  AdaptiveRouteResult* result = adaptive_router_route(
    router, embedding, 384, 0.5f
  );

  if (result) {
    printf("Selected: %s\n", result->selected_model);
    printf("Cluster: %d\n", result->cluster_id);

    // Free result
    adaptive_route_result_free(result);
  }

  // Cleanup
  adaptive_router_destroy(router);
  return 0;
}
```

**Simple routing (just get model ID):**

```c
char* model_id = adaptive_router_route_simple(router, embedding, 384, 0.5f);
printf("Selected: %s\n", model_id);
adaptive_string_free(model_id);
```

## API Reference

### Router Class (C++)

| Method | Description |
|--------|-------------|
| `static Router from_file(path)` | Load router from JSON profile file |
| `static Router from_json_string(json)` | Load router from JSON string |
| `static Router from_binary(path)` | Load router from MessagePack binary file |
| `RouteResponse route(request)` | Route using RouteRequest struct |
| `RouteResponse route(embedding_data, size, cost_bias)` | Route with raw float pointer |
| `RouteResponse route(embedding, cost_bias)` | Route with Eigen vector |
| `vector<string> get_supported_models()` | Get list of all model IDs |
| `int get_n_clusters()` | Get number of clusters |
| `int get_embedding_dim()` | Get expected embedding dimension |

### RouteRequest

```cpp
struct RouteRequest {
  std::span<const float> embedding;     // Pre-computed embedding
  float cost_bias = 0.5f;               // 0.0=accuracy, 1.0=cost
  std::vector<std::string> models;      // Optional model filter
};
```

### RouteResponse

| Field | Type | Description |
|-------|------|-------------|
| `selected_model` | string | Selected model ID (e.g., "openai/gpt-4") |
| `alternatives` | vector<string> | Alternative model IDs ranked by score |
| `cluster_id` | int | Assigned cluster ID (0 to K-1) |
| `cluster_distance` | float | Euclidean distance to cluster centroid |

### Python API (adaptive_core_ext)

```python
class Router:
    @staticmethod
    def from_file(path: str) -> Router: ...

    @staticmethod
    def from_json_string(json_str: str) -> Router: ...

    @staticmethod
    def from_binary(path: str) -> Router: ...

    def route(self, embedding: np.ndarray, cost_bias: float = 0.5) -> RouteResponse:
        """Route a single embedding. Zero-copy for float32 C-contiguous arrays."""
        ...

    def route_batch(self, embeddings: np.ndarray, cost_bias: float = 0.5) -> list[RouteResponse]:
        """Batch route multiple embeddings (N×D array). Zero-copy for float32 C-contiguous arrays."""
        ...

    def get_supported_models(self) -> list[str]: ...
    def get_n_clusters(self) -> int: ...
    def get_embedding_dim(self) -> int: ...

class RouteResponse:
    selected_model: str
    alternatives: list[str]
    cluster_id: int
    cluster_distance: float
```

### C API

See [`adaptive.h`](bindings/c/adaptive.h) for the complete C API reference.

Key functions:
- `adaptive_router_create(path)` - Create router from file
- `adaptive_router_route(router, embedding, size, cost_bias)` - Route request
- `adaptive_router_route_batch(router, embeddings, n_embeddings, size, cost_bias)` - Batch route
- `adaptive_router_route_simple(...)` - Simple routing (returns model ID only)
- `adaptive_router_destroy(router)` - Free router resources
- `adaptive_batch_route_result_free(result)` - Free batch result

## Performance

### Benchmarks (i7-12700K, single-threaded)

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Cluster assignment | ~0.1ms | 10,000 req/s |
| Model scoring | ~0.05ms | 20,000 req/s |
| Full `route()` call | ~0.2ms | 5,000 req/s |

**Python vs C++ Comparison:**

| Implementation | Latency | Throughput | Speedup |
|----------------|---------|------------|---------|
| Pure Python | ~2ms | 500 req/s | 1x |
| C++ Core | ~0.2ms | 5,000 req/s | **10x** |

### Memory Usage

- **Router instance**: 1-5 MB (depends on profile size)
- **Per-request overhead**: Zero heap allocations (stack-only)
- **Profile storage**: ~1 MB per 20 clusters with 10 models

### Optimization Features

- **Eigen vectorization**: SIMD instructions for matrix operations
- **Zero-copy**: Embedding data passed by reference/span
- **Stack allocation**: No per-request heap allocations
- **Move semantics**: Efficient resource transfer

## Profile Format

### JSON Structure

```json
{
  "metadata": {
    "n_clusters": 20,
    "embedding_model": "all-MiniLM-L6-v2",
    "silhouette_score": 0.42,
    "clustering": {
      "max_iter": 300,
      "random_state": 42,
      "algorithm": "lloyd",
      "normalization_strategy": "l2"
    },
    "routing": {
      "lambda_min": 0.0,
      "lambda_max": 2.0,
      "default_cost_preference": 0.5,
      "max_alternatives": 5
    }
  },
  "cluster_centers": [
    [0.1, 0.2, ...],  // Cluster 0 centroid (384D)
    [0.3, 0.4, ...],  // Cluster 1 centroid
    ...
  ],
  "models": [
    {
      "model_id": "openai/gpt-4",
      "provider": "openai",
      "model_name": "gpt-4",
      "error_rates": [0.05, 0.08, 0.06, ...],  // Per-cluster error rates
      "cost_per_1m_input_tokens": 30.0,
      "cost_per_1m_output_tokens": 60.0
    },
    ...
  ]
}
```

### MessagePack Binary Format

For faster loading, profiles can be saved in MessagePack format:

```bash
# Convert JSON to MessagePack (Python)
python scripts/convert_profile.py profile.json profile.msgpack
```

```cpp
// Load binary profile
auto router = Router::from_binary("profile.msgpack");
```

## Development

### Project Structure

```
adaptive_router_core/
├── include/              # Public C++ headers
│   ├── router.hpp        # Main router API
│   ├── cluster.hpp       # Cluster engine
│   ├── scorer.hpp        # Model scorer
│   ├── profile.hpp       # Profile loader
│   └── types.hpp         # Common types
├── src/                  # C++ implementation
│   ├── router.cpp
│   ├── cluster.cpp
│   ├── scorer.cpp
│   └── profile.cpp
├── bindings/             # Language bindings
│   ├── python/           # nanobind Python bindings
│   │   ├── CMakeLists.txt
│   │   └── adaptive_core_py.cpp
│   └── c/                # C FFI API
│       ├── adaptive.h
│       └── adaptive_c.cpp
├── tests/                # Google Test suite
│   ├── CMakeLists.txt
│   ├── test_cluster.cpp
│   ├── test_router.cpp
│   └── test_scorer.cpp
├── CMakeLists.txt        # Main CMake configuration
├── conanfile.py          # Conan dependencies
├── pyproject.toml        # Python package metadata
└── README.md
```

### Building with Tests

**Linux/macOS:**

```bash
# Configure with testing enabled
cmake --preset conan-release -DBUILD_TESTING=ON

# Build
cmake --build build/build/Release

# Run tests
ctest --test-dir build/build/Release --output-on-failure

# Run tests with verbose output
ctest --test-dir build/build/Release -V
```

**Windows:**

```bash
# Configure with testing enabled
cmake --preset conan-default -DBUILD_TESTING=ON

# Build
cmake --build build --config Release

# Run tests
ctest --test-dir build -C Release --output-on-failure

# Run tests with verbose output
ctest --test-dir build -C Release -V
```

### Code Quality

```bash
# Format code (requires clang-format)
find include src -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i

# Static analysis (requires clang-tidy)
cmake --build build/build/Release --target clang-tidy

# Generate compile_commands.json for IDE support (Linux/macOS)
cmake --preset conan-release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Generate compile_commands.json for IDE support (Windows)
cmake --preset conan-default -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Troubleshooting

### Common Issues

**"C++23 not supported" error**

Ensure you have a compatible compiler:
- GCC 12+: `g++ --version`
- Clang 15+: `clang++ --version`
- MSVC 2022+: Visual Studio 17.0+

**"Eigen not found" error**

Run Conan install first:
```bash
conan install . --output-folder=build --build=missing
```

**"Embedding dimension mismatch" error**

The router expects 384-dimensional embeddings (from all-MiniLM-L6-v2 model). Verify:
- Your embedding model matches the profile
- Embedding size is correct: `router.get_embedding_dim()` returns 384

**Python import error: "No module named 'adaptive_core_ext'"**

Ensure the package is installed:
```bash
pip install ./adaptive_router_core
# or
uv pip install ./adaptive_router_core
```

**Conan configuration issues on Windows**

Windows users may need to configure Conan profile:
```bash
conan profile detect --force
```

### Platform-Specific Notes

**Linux:**
- GCC 12+ available in Ubuntu 22.04+
- Install via: `sudo apt install g++-12 cmake`

**macOS:**
- Clang 15+ included in Xcode 14.3+
- Install via: `xcode-select --install`

**Windows:**
- Visual Studio 2022 (17.0+) includes C++23 support
- Ensure "Desktop development with C++" workload is installed

## Integration with Python Library

The C++ core is designed to work alongside the Python library:

1. **Python handles**:
   - Sentence transformer embedding computation
   - Profile training and management
   - High-level API and HTTP endpoints

2. **C++ core handles**:
   - Fast cluster assignment
   - Efficient model scoring
   - High-throughput routing

**Example integration:**

```python
from sentence_transformers import SentenceTransformer
from adaptive_core_ext import Router

# Python: Compute embeddings (GPU-accelerated)
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
embedding = model.encode("prompt").tolist()

# C++: Fast routing
router = Router.from_file("profile.json")
response = router.route(embedding, 0.5)
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/Egham-7/adaptive/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Egham-7/adaptive/discussions)
