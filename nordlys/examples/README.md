# Nordlys Examples

Learn how to use Nordlys with practical examples.

## Files

**test_imports.py** - Test all imports work
```bash
python examples/test_imports.py
```

**simple_example.py** - Complete working example
```bash
python examples/simple_example.py
```

## Quick Example

```python
from nordlys import Dataset, Trainer, Router, ModelConfig

# Define models
models = [
    ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ModelConfig(id="openai/gpt-4o-mini", cost_input=0.15, cost_output=0.6),
]

# Training dataset
dataset = Dataset.from_list([
    {
        "id": "1",
        "input": "Write code",
        "targets": {"openai/gpt-4": 1, "openai/gpt-4o-mini": 0},
    },
    {
        "id": "2",
        "input": "What is 2+2?",
        "targets": {"openai/gpt-4": 0, "openai/gpt-4o-mini": 1},
    },
])

# Train and route
checkpoint = Trainer(models=models).fit(dataset)
router = Router(checkpoint=checkpoint)
result = router.route("Write a function")
```

## Advanced Usage

### Custom Clustering

```python
from nordlys import Trainer
from nordlys.clustering import HDBSCANClusterer, KMeansClusterer

# Auto-discover clusters
trainer = Trainer(models=models, clusterer=HDBSCANClusterer(min_cluster_size=50))

# Fixed cluster count
trainer = Trainer(models=models, clusterer=KMeansClusterer(n_clusters=15))
```

### Dimensionality Reduction

```python
# Reducers are currently unsupported for checkpoint-compatible training.
# Use full embedding space for now.
```

### Inspect Clusters

```python
# Get metrics
metrics = router.get_metrics()
print(f"Silhouette: {metrics.silhouette_score:.3f}")

# Per-cluster details
for cluster in router.get_clusters():
    print(f"Cluster {cluster.cluster_id}: {cluster.size} samples")
```
