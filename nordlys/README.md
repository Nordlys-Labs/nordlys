# Nordlys

Smart LLM model router. Picks the best model for each prompt based on cost and quality.

## Install

```bash
# CPU only
uv pip install nordlys[cpu]

# CUDA 12 (Linux only)
uv pip install nordlys[cu12]
```

## Quick Start

```python
from nordlys import Nordlys, ModelConfig
import pandas as pd

# 1. Define your models
models = [
    ModelConfig(id="openai/gpt-4", cost_input=30.0, cost_output=60.0),
    ModelConfig(id="openai/gpt-3.5-turbo", cost_input=0.5, cost_output=1.5),
]

# 2. Training data: questions + accuracy scores per model
df = pd.DataFrame({
    "questions": ["Write code", "What is 2+2?", "Explain quantum physics"],
    "openai/gpt-4": [0.95, 0.99, 0.92],
    "openai/gpt-3.5-turbo": [0.70, 0.99, 0.60],
})

# 3. Fit and route
router = Nordlys(models=models)
router.fit(df)

result = router.route("Write a sorting algorithm")
print(result.model_id)        # Best model for this prompt
print(result.alternatives)    # Ranked alternatives
print(result.cluster_id)      # Which cluster it matched
```

## How It Works

1. **Embeds** prompts using sentence-transformers (default: `all-MiniLM-L6-v2`)
2. **Optionally reduces** dimensionality with UMAP or PCA
3. **Clusters** similar prompts together (K-Means, HDBSCAN, GMM, Agglomerative, or Spectral)
4. **Learns** which model performs best per cluster
5. **Routes** new prompts to the optimal model via the C++ core engine

## Routing

```python
# Route a single prompt
result = router.route("Explain quantum physics")
# result.model_id          -> "openai/gpt-4"
# result.cluster_id        -> 3
# result.cluster_distance  -> 0.42
# result.alternatives      -> ["openai/gpt-3.5-turbo"]

# Route with a model filter (only consider specific models)
result = router.route("Write code", models=["openai/gpt-4"])

# Route a batch of prompts
results = router.route_batch(["Write code", "What is 2+2?", "Translate this"])
```

## Custom Clustering & Reduction

```python
from nordlys import Nordlys, ModelConfig
from nordlys.clustering import HDBSCANClusterer, SpectralClusterer
from nordlys.reduction import UMAPReducer, PCAReducer

# Use UMAP + HDBSCAN
router = Nordlys(
    models=models,
    umap_model=UMAPReducer(n_components=10),
    cluster_model=HDBSCANClusterer(min_cluster_size=5),
)

# Or PCA + Spectral clustering
router = Nordlys(
    models=models,
    umap_model=PCAReducer(n_components=50),
    cluster_model=SpectralClusterer(n_clusters=15),
)

# Custom embedding model
router = Nordlys(
    models=models,
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    device="cuda",  # Use GPU for embeddings and C++ core
)
```

### Available Clusterers

| Clusterer | Description |
|---|---|
| `KMeansClusterer` | K-Means (default) |
| `HDBSCANClusterer` | Density-based, auto-discovers cluster count |
| `GMMClusterer` | Gaussian Mixture Model |
| `AgglomerativeClusterer` | Hierarchical agglomerative |
| `SpectralClusterer` | Graph-based spectral clustering |

### Available Reducers

| Reducer | Description |
|---|---|
| `UMAPReducer` | Non-linear dimensionality reduction |
| `PCAReducer` | Linear dimensionality reduction |

## Save & Load

```python
# Save as JSON
router.save("router.json")

# Save as MessagePack (smaller, faster)
router.save("router.msgpack")

# Load
loaded = Nordlys.load("router.json")
loaded = Nordlys.load("router.msgpack")

# Load with device selection
loaded = Nordlys.load("router.json", device="cuda")

# Load with overridden model costs
loaded = Nordlys.load("router.json", models=[
    ModelConfig(id="openai/gpt-4", cost_input=25.0, cost_output=50.0),
])
```

## Introspection

```python
# Cluster info
info = router.get_cluster_info(0)
# info.cluster_id, info.size, info.centroid, info.model_accuracies

# All clusters
clusters = router.get_clusters()

# Clustering metrics
metrics = router.get_metrics()
# metrics.silhouette_score, metrics.n_clusters, metrics.inertia

# sklearn-style fitted attributes
router.centroids_          # (n_clusters, n_features)
router.labels_             # (n_samples,)
router.embeddings_         # (n_samples, embedding_dim)
router.model_accuracies_   # {cluster_id: {model_id: accuracy}}
```

## Links

- [Docs](https://docs.nordlyslabs.com)
- [Issues](https://github.com/Nordlys-Labs/nordlys/issues)

## Citation

This project is inspired by the Universal Router approach:

```bibtex
@article{universalrouter2025,
  title={Universal Router: Foundation Model Routing for Arbitrary Tasks},
  author={},
  journal={arXiv preprint arXiv:2502.08773},
  year={2025},
  url={https://arxiv.org/pdf/2502.08773}
}
```

**Paper**: [Universal Router: Foundation Model Routing for Arbitrary Tasks](https://arxiv.org/pdf/2502.08773)
