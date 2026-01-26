# Clustering Module

K-means clustering engine with CPU and CUDA backends for embedding-to-cluster assignment.

## Overview

The clustering module provides:
- **CPU backend** - Using USearch/SimSIMD for SIMD-accelerated distance computation
- **CUDA backend** - GPU-accelerated clustering for high throughput (optional)
- **Batch operations** - Efficient processing of multiple embeddings

## Headers

```cpp
#include <nordlys/clustering/cluster.hpp>        // ClusterEngine class
#include <nordlys/clustering/embedding_view.hpp> // EmbeddingView type

// CUDA-specific (when NORDLYS_HAS_CUDA is defined)
#include <nordlys/clustering/cuda/memory.cuh>    // CUDA memory management
#include <nordlys/clustering/cuda/distance.cuh>  // Distance kernels
#include <nordlys/clustering/cuda/reduce.cuh>    // Reduction kernels
#include <nordlys/clustering/cuda/common.cuh>    // Common CUDA utilities
```

## CMake

```cmake
target_link_libraries(your_target PRIVATE Nordlys::Clustering)
```

## Usage

### Basic Clustering

```cpp
#include <nordlys/clustering/cluster.hpp>

// Create cluster engine (auto-selects best backend)
ClusterEngine engine;

// Or explicitly select backend
ClusterEngine cpu_engine(Device::CPU);
ClusterEngine cuda_engine(Device::CUDA);  // Requires CUDA build

// Load centroids from checkpoint
engine.load_centroids(centroids_matrix);

// Assign single embedding to cluster
std::vector<float> embedding(128);
auto [cluster_id, distance] = engine.assign(embedding);

// Batch assignment
std::vector<std::vector<float>> embeddings = /* ... */;
auto results = engine.assign_batch(embeddings);
```

### Embedding View

```cpp
#include <nordlys/clustering/embedding_view.hpp>

// Non-owning view into embedding data
float* data = /* ... */;
size_t dim = 128;
EmbeddingView view(data, dim);
```

## CUDA Backend

When built with `NORDLYS_ENABLE_CUDA=ON`, the clustering module includes:

- **Fused distance kernels** - Compute L2 distance in single pass
- **Warp-level reductions** - Fast argmin using shuffle instructions
- **CUDA graphs** - Captured kernel sequences for minimal launch overhead
- **Pinned memory** - Zero-copy transfers for batch operations

### CUDA Memory Management

```cpp
#include <nordlys/clustering/cuda/memory.cuh>

// Device memory with RAII
DevicePtr<float> d_data(1024);

// Pinned host memory for fast transfers
PinnedPtr<float> h_data(1024);

// Copy data
cudaMemcpy(d_data.get(), h_data.get(), 1024 * sizeof(float), cudaMemcpyHostToDevice);
```

## Performance

| Backend | Single (128-dim) | Batch 1000 (128-dim) |
|---------|------------------|----------------------|
| CPU     | ~5us             | ~500us               |
| CUDA    | ~50us (overhead) | ~50us (amortized)    |

CUDA provides significant speedup for batch operations where kernel launch overhead is amortized.

## Dependencies

- `Nordlys::Common` - Matrix and device types
- `USearch` - CPU vector search
- `SimSIMD` - SIMD distance computations
- `CUDA Toolkit` - GPU support (optional)
