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
from nordlys import Nordlys, ModelConfig
import pandas as pd

# Define models
models = [
    ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ModelConfig(id="openai/gpt-3.5-turbo", cost_input=0.5, cost_output=1.5),
]

# Training data
df = pd.DataFrame({
    "questions": ["Write code", "What is 2+2?"],
    "openai/gpt-4": [0.95, 0.99],
    "openai/gpt-3.5-turbo": [0.70, 0.99],
})

# Fit and route
router = Nordlys(models=models)
router.fit(df)
result = router.route("Write a function", cost_bias=0.5)
```

## Advanced Usage

### Custom Clustering

```python
from nordlys.clustering import HDBSCANClusterer, GMMClusterer

# Auto-discover clusters
model = Nordlys(models, cluster_model=HDBSCANClusterer(min_cluster_size=50))

# Gaussian mixture
model = Nordlys(models, cluster_model=GMMClusterer(n_components=15))
```

### Dimensionality Reduction

```python
from nordlys.reduction import UMAPReducer, PCAReducer

# With UMAP
model = Nordlys(models, umap_model=UMAPReducer(n_components=3))

# With PCA
model = Nordlys(models, umap_model=PCAReducer(n_components=50))
```

### Inspect Clusters

```python
# Get metrics
metrics = model.get_metrics()
print(f"Silhouette: {metrics.silhouette_score:.3f}")

# Per-cluster details
for cluster in model.get_clusters():
    print(f"Cluster {cluster.cluster_id}: {cluster.size} samples")
```
