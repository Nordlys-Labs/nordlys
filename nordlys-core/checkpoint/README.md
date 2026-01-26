# Checkpoint Module

Checkpoint serialization and deserialization for model routing profiles.

## Overview

The checkpoint module handles loading and saving routing profiles in:
- **JSON** - Human-readable format for debugging
- **MessagePack** - Binary format for production (faster, smaller)

## Headers

```cpp
#include <nordlys/checkpoint/checkpoint.hpp>  // NordlysCheckpoint class
#include <nordlys/checkpoint/cache.hpp>       // Checkpoint caching utilities
```

## CMake

```cmake
target_link_libraries(your_target PRIVATE Nordlys::Checkpoint)
```

## Usage

### Loading Checkpoints

```cpp
#include <nordlys/checkpoint/checkpoint.hpp>

// From JSON file
auto checkpoint = NordlysCheckpoint::from_json_file("profile.json");

// From MessagePack file
auto checkpoint = NordlysCheckpoint::from_msgpack_file("profile.msgpack");

// From JSON string
std::string json_str = /* ... */;
auto checkpoint = NordlysCheckpoint::from_json_string(json_str);
```

### Accessing Checkpoint Data

```cpp
// Get embedding dimension
size_t dim = checkpoint.embedding_dim();

// Get number of clusters
size_t n_clusters = checkpoint.n_clusters();

// Get centroids matrix
const auto& centroids = checkpoint.centroids();

// Get model features per cluster
const auto& models = checkpoint.models();
```

### Saving Checkpoints

```cpp
// To JSON file
checkpoint.to_json_file("profile.json");

// To MessagePack file
checkpoint.to_msgpack_file("profile.msgpack");
```

## Checkpoint Format

```json
{
  "embedding_dim": 128,
  "n_clusters": 10,
  "centroids": [[...], [...], ...],
  "models": [
    {
      "name": "gpt-4",
      "provider": "openai",
      "error_rates": [0.1, 0.2, ...],
      "cost_per_input_token": 0.00003,
      "cost_per_output_token": 0.00006
    }
  ]
}
```

## Dependencies

- `Nordlys::Common` - Matrix types
- `Nordlys::Scoring` - ModelFeatures type
- `nlohmann_json` - JSON parsing
- `msgpack-cxx` - MessagePack serialization
- `simdjson` - Fast JSON parsing
