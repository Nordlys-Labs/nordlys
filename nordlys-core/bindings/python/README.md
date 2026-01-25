# Python Bindings

Python bindings for the Nordlys C++ core using nanobind.

## Overview

Provides native Python APIs for the Nordlys routing engine with:
- Type stubs for IDE support
- NumPy array compatibility
- Native Python types (lists, dicts, strings)
- Used by the main `nordlys` Python package

## Installation

```bash
# Install from source using uv
uv sync --group test
uv pip install -e .

# Or via the main nordlys package
uv pip install nordlys
```

## Usage

```python
from nordlys_core import Nordlys32, NordlysCheckpoint

# Load checkpoint
checkpoint = NordlysCheckpoint.from_json_file("profile.json")
router = Nordlys32.from_checkpoint(checkpoint)

# Route embedding
embedding = [0.1, 0.2, 0.3, ...]  # List or NumPy array
result = router.route(embedding)

# Access results
print(f"Selected: {result.selected_model}")
print(f"Cluster: {result.cluster_id}")
print(f"Alternatives: {result.alternatives}")
```

## API Reference

### `Nordlys32` / `Nordlys64`
Main router class for float32/float64 precision.

**Methods:**
- `from_checkpoint(checkpoint)` - Create router from checkpoint
- `route(embedding, models=None)` - Route single embedding
- `route_batch(embeddings, models=None)` - Route batch of embeddings
- `get_supported_models()` - Get list of available models
- `get_n_clusters()` - Get number of clusters
- `get_embedding_dim()` - Get embedding dimension

### `NordlysCheckpoint`
Routing profile data structure.

**Methods:**
- `from_json_file(path)` - Load from JSON file
- `from_json_string(json)` - Load from JSON string
- `to_json()` - Serialize to JSON string

### `RouteResult32` / `RouteResult64`
Routing result with selected model and metadata.

**Attributes:**
- `selected_model` - Selected model ID (str)
- `alternatives` - List of alternative model IDs
- `cluster_id` - Assigned cluster ID (int)
- `cluster_distance` - Distance to cluster center (float)

## Testing

```bash
# Run Python binding tests with uv
uv run pytest tests/

# Or with verbose output
uv run pytest tests/ -v

# Or using ctest (from build directory)
ctest -R python_bindings --output-on-failure
```

## See Also

- [Main README](../../README.md) - Build and usage guide
- [Core Library](../../core/README.md) - C++ implementation
- [Bindings Overview](../README.md) - All language bindings
