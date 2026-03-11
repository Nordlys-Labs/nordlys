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

Reference measurements from the current `Release` benchmark build on an **Intel Core i9-10900K + RTX 3070**:

| Benchmark | Result |
|-----------|--------|
| CPU single assign, `100c/1536d` | ~31 us |
| CUDA single assign, GPU-resident, `100c/1536d` | ~38 us |
| CUDA batch assign, GPU-resident, `64/100c/1536d` | ~30 us (~2.13M embeddings/sec) |
| CUDA batch assign, GPU-resident, `256/100c/1536d` | ~78 us (~3.27M embeddings/sec) |

Single-query workloads are often still faster on CPU. CUDA becomes attractive when embeddings are already GPU-resident and processed in batches large enough to amortize launch overhead.

## Dependencies

- `Nordlys::Common` - Matrix and device types
- `USearch` - CPU vector search
- `SimSIMD` - SIMD distance computations
- `CUDA Toolkit` - GPU support (optional)
