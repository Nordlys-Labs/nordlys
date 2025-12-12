# Adaptive Router Core - Python Bindings

This package provides Python bindings for the Adaptive Router Core C++ library, enabling high-performance model routing based on cluster-based selection.

## Installation

```bash
pip install .
```

## Usage

```python
from adaptive_core_ext import Router
import numpy as np

# Create router from JSON profile
router = Router.from_json_file("profile.json")

# Route an embedding
embedding = np.random.randn(384).astype(np.float32)
response = router.route(embedding, cost_bias=0.5)

print(f"Selected model: {response.selected_model}")
print(f"Cluster ID: {response.cluster_id}")
print(f"Alternatives: {response.alternatives}")
```

## API Reference

### Router Class

#### Factory Methods
- `Router.from_json_file(path)` - Load from JSON file
- `Router.from_json_string(json_str)` - Load from JSON string
- `Router.from_msgpack_file(path)` - Load from MessagePack file

#### Routing Methods
- `route(embedding, cost_bias=0.5)` - Route single embedding
- `route_batch(embeddings, cost_bias=0.5)` - Route multiple embeddings

#### Introspection
- `get_supported_models()` - Get list of supported model IDs
- `get_n_clusters()` - Get number of clusters
- `get_embedding_dim()` - Get expected embedding dimension

### RouteResponse Class
- `selected_model` - Selected model ID (str)
- `alternatives` - List of alternative model IDs (list[str])
- `cluster_id` - Assigned cluster ID (int)
- `cluster_distance` - Distance to cluster centroid (float)

## Performance

- **Embedding computation**: 10-50ms (Python/SentenceTransformer)
- **Routing**: ~0.15ms (C++ core)
- **Total latency**: 10-50ms end-to-end

## Requirements

- Python 3.11+
- NumPy
- Compatible C++ compiler (GCC 9+, Clang 10+, MSVC 2019+)

## Development

For development with faster iteration:

```bash
# Build core once
cd ../..
cmake -B build -DADAPTIVE_BUILD_TESTS=ON
cmake --build build

# Install bindings in editable mode
cd bindings/python
pip install --no-build-isolation -e .
pytest tests/
```