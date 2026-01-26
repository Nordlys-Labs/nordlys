# Routing Module

High-level routing API that combines clustering, scoring, and checkpoint loading.

## Overview

The routing module provides the main `Nordlys` class - the primary interface for model routing:

1. Load a checkpoint with cluster centroids and model features
2. Assign input embeddings to clusters
3. Score and rank models for the assigned cluster
4. Return the best model with alternatives

## Headers

```cpp
#include <nordlys/routing/nordlys.hpp>
```

## CMake

```cmake
target_link_libraries(your_target PRIVATE Nordlys::Routing)

# Or use the unified interface that links all modules
target_link_libraries(your_target PRIVATE Nordlys::Core)
```

## Usage

### Basic Routing

```cpp
#include <nordlys/routing/nordlys.hpp>

// Load from checkpoint file
auto router = Nordlys::from_json_file("checkpoint.json");

// Or from checkpoint object
auto checkpoint = NordlysCheckpoint::from_json_file("checkpoint.json");
auto router = Nordlys::from_checkpoint(std::move(checkpoint));

// Route an embedding
std::vector<float> embedding(128);
float cost_bias = 0.5f;  // 0.0 = prefer accuracy, 1.0 = prefer cost

auto result = router.route(embedding, cost_bias);

// Access result
std::cout << "Selected: " << result.selected_model << "\n";
std::cout << "Cluster: " << result.cluster_id << "\n";
std::cout << "Distance: " << result.cluster_distance << "\n";

// Get alternatives
for (const auto& alt : result.alternatives) {
    std::cout << "Alt: " << alt.model_name << " (score: " << alt.score << ")\n";
}
```

### Batch Routing

```cpp
// Route multiple embeddings efficiently
std::vector<std::vector<float>> embeddings = /* ... */;
auto results = router.route_batch(embeddings, cost_bias);
```

### Model Filtering

```cpp
// Only consider specific models
std::unordered_set<std::string> allowed = {"gpt-4", "claude-3"};
auto result = router.route(embedding, cost_bias, allowed);
```

### Router Properties

```cpp
// Get embedding dimension (for validation)
size_t dim = router.embedding_dim();

// Get number of clusters
size_t n_clusters = router.n_clusters();

// Get supported models
auto models = router.supported_models();
```

## Thread Safety

The `Nordlys` router is **thread-safe** for concurrent routing calls. Multiple threads can call `route()` and `route_batch()` simultaneously without external synchronization.

## Error Handling

```cpp
try {
    auto router = Nordlys::from_json_file("missing.json");
} catch (const std::runtime_error& e) {
    // File not found, invalid JSON, etc.
}

try {
    std::vector<float> wrong_dim(64);  // Expected 128
    auto result = router.route(wrong_dim, 0.5f);
} catch (const std::invalid_argument& e) {
    // Dimension mismatch
}
```

## Dependencies

- `Nordlys::Clustering` - Embedding-to-cluster assignment
- `Nordlys::Scoring` - Model ranking
- `Nordlys::Checkpoint` - Profile loading
- `Nordlys::Common` - Shared types
