# Core Library

The core C++ library implementing Nordlys routing algorithms and data structures.

## Overview

The core library provides:
- **Routing Engine**: Model selection based on embeddings and cost optimization
- **Clustering**: K-means cluster assignment with CPU and CUDA backends
- **Checkpoint Management**: Loading and validation of routing profiles
- **Model Scoring**: Cost-accuracy trade-off optimization

## Components

### `nordlys.hpp`
Main routing engine that orchestrates clustering and model selection.

**Key Classes:**
- `Nordlys<Scalar>` - Main router class (float32/float64 precision)
- `RouteResult<Scalar>` - Routing result with selected model and alternatives

### `cluster.hpp`
Cluster assignment engine for embedding-to-cluster mapping.

**Key Classes:**
- `ClusterEngine<Scalar>` - Cluster assignment with CPU/CUDA backends
- `ClusterBackendType` - Backend selection (CPU, CUDA)

### `scorer.hpp`
Model scoring and ranking based on error rates and costs.

**Key Classes:**
- `ModelScorer` - Scores and ranks models for a cluster
- `ModelScore` - Score result with model ID and metrics
- `ModelFeatures` - Model metadata (costs, error rates)

### `checkpoint.hpp`
Checkpoint loading and validation.

**Key Classes:**
- `NordlysCheckpoint` - Routing profile data structure
- JSON parsing and validation

### `result.hpp`
Error handling and result types.

**Key Classes:**
- `Result<T, E>` - Rust-like result type for error handling

## Usage

```cpp
#include <nordlys_core/nordlys.hpp>

// Load checkpoint
auto checkpoint = NordlysCheckpoint::from_json_file("profile.json");
auto router = Nordlys32::from_checkpoint(std::move(checkpoint));

// Route embedding
float embedding[] = {0.1f, 0.2f, ...};
auto result = router.route(embedding, embedding_size, 0.5f);

// Use result
std::cout << "Selected: " << result.selected_model << std::endl;
```

## Performance

- **Single routing**: ~50-500μs (depends on profile size)
- **Batch routing**: ~0.1-1ms per 100 embeddings
- **Memory**: ~10-50MB (depends on profile size)
- **Thread-safe**: Safe for concurrent routing operations

## Testing

```bash
# Run all core tests
ctest -R test_nordlys_core

# Run specific test suites
ctest -R test_scorer
ctest -R test_cluster
ctest -R test_checkpoint
```

## See Also

- [Main README](../README.md) - Build and usage guide
- [Bindings](../bindings/README.md) - Language bindings
