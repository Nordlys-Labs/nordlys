# Adaptive Router: C++ Core Migration Plan

## Executive Summary

Migrate the adaptive_router **inference core only** from Python to C++ to enable:
1. **Faster inference** via C++ with ONNX Runtime + CUDA
2. **FFI bindings** for multiple languages (Python, Go, Node.js in future)
3. **Single source of truth** for routing logic across all platforms

**What stays in Python:**
- FastAPI server (Modal deployment unchanged)
- Training pipeline (PyTorch, sklearn, DeepEval)
- Profile generation and export (JSON + binary format for C++ core)

**What moves to C++:**
- Embedding extraction (ONNX Runtime)
- K-means cluster assignment
- Model scoring algorithm

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Modal Deployment (Python)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              FastAPI Server (unchanged)              │   │
│  │                   /select-model                      │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Python Wrapper (adaptive_router)           │   │
│  │         router.py calls C++ core via bindings        │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │ nanobind                         │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              C++ Core (libadaptive_core)             │   │
│  │  • ONNX Runtime (embeddings, CUDA)                  │   │
│  │  • Eigen (K-means, matrix ops)                      │   │
│  │  • Scoring algorithm                                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                Training Pipeline (Pure Python)               │
│  • Trainer class (PyTorch, sklearn, DeepEval)               │
│  • Exports RouterProfile as JSON (debugging)                │
│  • Exports RouterProfile as MessagePack binary (production) │
│  • C++ core loads binary format for fast startup            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Future: Direct FFI from Other Languages         │
│  • Go proxy can call C++ core directly (cgo)                │
│  • Node.js can use N-API bindings                           │
│  • Same C++ core, different language wrappers               │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Server | **Python/FastAPI (Modal)** | Keep existing deployment, just swap inference |
| Training | **Pure Python** | Uses PyTorch, sklearn, DeepEval - no need to port |
| C++ Core | **Library only** | Exposed via Python bindings (nanobind) |
| GPU Support | **CUDA via ONNX Runtime** | T4 GPU on Modal |
| Profile Format | **JSON + Binary** | JSON for dev, MessagePack for production |
| Go Integration | **HTTP API initially** | Later: direct cgo to C++ core |

---

## C++ Tech Stack

### Dependencies

```cmake
# Core ML
onnxruntime         >= 1.17.0    # ONNX Runtime C++ API (official)
eigen3              >= 3.4       # Matrix operations
nlohmann_json       >= 3.11      # JSON parsing (header-only)
msgpack-cxx         >= 6.0       # Binary profile format

# Tokenization
tokenizers-cpp      >= 0.2       # HuggingFace tokenizers C++ bindings

# FFI Bindings
nanobind            >= 1.8       # Python bindings (4x faster than pybind11)

# Build & Package
cmake               >= 3.20
conan               >= 2.0       # Package manager
```

### Project Structure

```
adaptive-router-core/
├── CMakeLists.txt
├── conanfile.txt
├── include/
│   └── adaptive/
│       ├── router.hpp              # Main public API
│       ├── embeddings.hpp          # ONNX embedding extraction
│       ├── cluster.hpp             # K-means cluster assignment
│       ├── scorer.hpp              # Model scoring engine
│       ├── profile.hpp             # Profile data structures
│       └── types.hpp               # Common types
├── src/
│   ├── router.cpp
│   ├── embeddings.cpp
│   ├── cluster.cpp
│   ├── scorer.cpp
│   └── profile.cpp
├── bindings/
│   ├── c/                          # C API for Go/others
│   │   ├── adaptive.h
│   │   └── adaptive_c.cpp
│   └── python/                     # nanobind Python module
│       ├── CMakeLists.txt
│       └── adaptive_py.cpp
├── models/
│   └── all-MiniLM-L6-v2.onnx      # Pre-exported embedding model
├── tests/
│   ├── test_embeddings.cpp
│   ├── test_cluster.cpp
│   ├── test_scorer.cpp
│   └── test_router.cpp
└── scripts/
    ├── export_onnx.py              # Export sentence-transformer to ONNX
    └── validate_embeddings.py      # Validate C++ vs Python embeddings
```

---

## Component Mapping: Python → C++

### 1. Embedding Extraction

**Python** (`cluster_engine.py:191-214`):
```python
embeddings = self.embedding_model.encode(
    texts, batch_size=32, normalize_embeddings=False
)
embeddings_normalized = normalize(embeddings, norm='l2')
```

**C++** (`embeddings.hpp`):
```cpp
class EmbeddingModel {
public:
    explicit EmbeddingModel(const std::string& model_path);

    // Single text → 384D embedding
    Eigen::VectorXf embed(const std::string& text);

    // Batch texts → Nx384 matrix
    Eigen::MatrixXf embed_batch(const std::vector<std::string>& texts);

private:
    Ort::Env env_;
    Ort::Session session_;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

    // L2 normalize
    Eigen::VectorXf normalize_l2(const Eigen::VectorXf& vec);
};
```

### 2. Cluster Assignment

**Python** (`cluster_engine.py:324-362`):
```python
def assign_single(self, text: str) -> tuple[int, float]:
    embeddings = self._extract_embeddings([text])
    embeddings_normalized = self._normalize_features(embeddings, norm='l2')
    cluster_id = int(self.kmeans.predict(embeddings_normalized)[0])
    distances = self.kmeans.transform(embeddings_normalized)[0]
    return cluster_id, float(distances[cluster_id])
```

**C++** (`cluster.hpp`):
```cpp
class ClusterEngine {
public:
    void load_centroids(const Eigen::MatrixXf& centers);  // K x 384

    // Returns (cluster_id, distance)
    std::pair<int, float> assign(const Eigen::VectorXf& embedding);

private:
    Eigen::MatrixXf centroids_;  // K x D
    int n_clusters_;
    int feature_dim_;

    // Squared Euclidean distance to each centroid
    Eigen::VectorXf compute_distances(const Eigen::VectorXf& embedding);
};
```

### 3. Model Scoring

**Python** (`router.py:174-204`):
```python
def _score_models(self, models_to_score, cluster_id, lambda_param):
    scored_models = []
    for model_id, features in models_to_score.items():
        error_rate = features.error_rates[cluster_id]
        normalized_cost = self._normalize_cost(features.cost_per_1m_tokens)
        score = error_rate + lambda_param * normalized_cost
        heapq.heappush(scored_models, (score, model_id, ModelScore(...)))
    return heapq.nsmallest(len(scored_models), scored_models)
```

**C++** (`scorer.hpp`):
```cpp
struct ModelScore {
    std::string model_id;
    float score;
    float error_rate;
    float accuracy;
    float cost;
    float normalized_cost;
};

class ModelScorer {
public:
    void load_models(const std::vector<ModelFeatures>& models);
    void set_cost_range(float min_cost, float max_cost);
    void set_lambda_params(float lambda_min, float lambda_max);

    // Main scoring
    std::vector<ModelScore> score_models(
        int cluster_id,
        float cost_bias,
        const std::vector<std::string>& filter = {}
    );

private:
    std::unordered_map<std::string, ModelFeatures> models_;
    float min_cost_, max_cost_, cost_range_;
    float lambda_min_, lambda_max_;

    float calculate_lambda(float cost_bias);
    float normalize_cost(float cost);
};
```

### 4. Main Router

**C++** (`router.hpp`):
```cpp
struct RouteRequest {
    std::string prompt;
    float cost_bias = 0.5f;
    std::vector<std::string> models;  // Optional filter
};

struct RouteResponse {
    std::string selected_model;
    std::vector<std::string> alternatives;
};

class Router {
public:
    // Factory methods
    static Router from_file(const std::string& profile_path);
    static Router from_json(const std::string& json_str);
    static Router from_binary(const std::string& path);

    // Main API
    RouteResponse route(const RouteRequest& request);

    // Simple API
    std::string route(const std::string& prompt, float cost_bias = 0.5f);

    // Introspection
    std::vector<std::string> get_supported_models() const;
    int get_n_clusters() const;

private:
    EmbeddingModel embeddings_;
    ClusterEngine clusters_;
    ModelScorer scorer_;
    ProfileMetadata metadata_;
};
```

### 5. Profile Data Structures

**C++** (`profile.hpp`):
```cpp
struct ModelFeatures {
    std::string model_id;
    std::string provider;
    std::string model_name;
    std::vector<float> error_rates;  // K values
    float cost_per_1m_input_tokens;
    float cost_per_1m_output_tokens;

    float cost_per_1m_tokens() const {
        return (cost_per_1m_input_tokens + cost_per_1m_output_tokens) / 2.0f;
    }
};

struct ClusteringConfig {
    int max_iter = 300;
    int random_state = 42;
    int n_init = 10;
    std::string algorithm = "lloyd";
    std::string normalization_strategy = "l2";
};

struct RoutingConfig {
    float lambda_min = 0.0f;
    float lambda_max = 2.0f;
    float default_cost_preference = 0.5f;
};

struct ProfileMetadata {
    int n_clusters;
    std::string embedding_model;
    float silhouette_score;
    ClusteringConfig clustering;
    RoutingConfig routing;
};

struct RouterProfile {
    Eigen::MatrixXf cluster_centers;  // K x D
    std::vector<ModelFeatures> models;
    ProfileMetadata metadata;

    static RouterProfile from_json(const std::string& path);
    static RouterProfile from_binary(const std::string& path);
};
```

---

## FFI Bindings

### C API (`bindings/c/adaptive.h`)

```c
#ifndef ADAPTIVE_ROUTER_H
#define ADAPTIVE_ROUTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct AdaptiveRouter AdaptiveRouter;

// Lifecycle
AdaptiveRouter* adaptive_router_create(const char* profile_path);
AdaptiveRouter* adaptive_router_create_from_json(const char* json_str);
void adaptive_router_destroy(AdaptiveRouter* router);

// Routing
typedef struct {
    const char* selected_model;
    const char** alternatives;
    size_t alternatives_count;
} AdaptiveRouteResult;

AdaptiveRouteResult* adaptive_router_route(
    AdaptiveRouter* router,
    const char* prompt,
    float cost_bias
);

// Simple routing (returns model_id, caller must free)
char* adaptive_router_route_simple(
    AdaptiveRouter* router,
    const char* prompt,
    float cost_bias
);

// Cleanup
void adaptive_route_result_free(AdaptiveRouteResult* result);
void adaptive_string_free(char* str);

// Introspection
size_t adaptive_router_get_n_clusters(AdaptiveRouter* router);
const char** adaptive_router_get_supported_models(AdaptiveRouter* router, size_t* count);

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_ROUTER_H
```

### Python Bindings (`bindings/python/adaptive_py.cpp`)

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "adaptive/router.hpp"

namespace nb = nanobind;

NB_MODULE(adaptive_core, m) {
    m.doc() = "Adaptive Router C++ Core";

    nb::class_<adaptive::RouteResponse>(m, "RouteResponse")
        .def_ro("selected_model", &adaptive::RouteResponse::selected_model)
        .def_ro("alternatives", &adaptive::RouteResponse::alternatives);

    nb::class_<adaptive::Router>(m, "Router")
        .def_static("from_file", &adaptive::Router::from_file)
        .def_static("from_json", &adaptive::Router::from_json)
        .def_static("from_binary", &adaptive::Router::from_binary)
        .def("route", nb::overload_cast<const std::string&, float>(
            &adaptive::Router::route),
            nb::arg("prompt"), nb::arg("cost_bias") = 0.5f)
        .def("get_supported_models", &adaptive::Router::get_supported_models)
        .def("get_n_clusters", &adaptive::Router::get_n_clusters);
}
```

---

## Profile Format

### JSON Format (Human-readable, development)

```json
{
    "cluster_centers": {
        "n_clusters": 20,
        "feature_dim": 384,
        "cluster_centers": [[...], [...], ...]
    },
    "models": [
        {
            "provider": "openai",
            "model_name": "gpt-4",
            "cost_per_1m_input_tokens": 30.0,
            "cost_per_1m_output_tokens": 60.0,
            "error_rates": [0.05, 0.08, 0.03, ...]
        }
    ],
    "metadata": {
        "n_clusters": 20,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "silhouette_score": 0.42,
        "clustering": {
            "max_iter": 300,
            "random_state": 42,
            "n_init": 10,
            "algorithm": "lloyd",
            "normalization_strategy": "l2"
        },
        "routing": {
            "lambda_min": 0.0,
            "lambda_max": 2.0,
            "default_cost_preference": 0.5
        }
    }
}
```

### Binary Format (Production, faster loading)

**Python Trainer Export** (`savers/binary.py`):
```python
import msgpack
import numpy as np

class BinaryProfileSaver:
    """Saves RouterProfile in binary MessagePack format for C++ core."""

    def save(self, profile: RouterProfile, output_path: str):
        """Export profile to MessagePack binary format."""
        centers_array = np.array(
            profile.cluster_centers.cluster_centers,
            dtype=np.float32
        )

        binary_data = {
            "cluster_centers": {
                "n_clusters": profile.cluster_centers.n_clusters,
                "feature_dim": profile.cluster_centers.feature_dim,
                "data": centers_array.tobytes(),
            },
            "models": [
                {
                    "provider": m.provider,
                    "model_name": m.model_name,
                    "cost_per_1m_input_tokens": m.cost_per_1m_input_tokens,
                    "cost_per_1m_output_tokens": m.cost_per_1m_output_tokens,
                    "error_rates": m.error_rates,
                }
                for m in profile.models
            ],
            "metadata": {
                "n_clusters": profile.metadata.n_clusters,
                "embedding_model": profile.metadata.embedding_model,
                "silhouette_score": profile.metadata.silhouette_score,
                "clustering": profile.metadata.clustering.model_dump(),
                "routing": profile.metadata.routing.model_dump(),
            },
        }

        with open(output_path, "wb") as f:
            msgpack.pack(binary_data, f)
```

**C++ Binary Loader** (`profile.cpp`):
```cpp
#include <msgpack.hpp>
#include <fstream>

RouterProfile RouterProfile::from_binary(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    std::string buffer((std::istreambuf_iterator<char>(ifs)),
                       std::istreambuf_iterator<char>());

    auto obj = msgpack::unpack(buffer.data(), buffer.size()).get();
    auto map = obj.as<std::map<std::string, msgpack::object>>();

    // Deserialize cluster centers from raw bytes
    auto centers_map = map["cluster_centers"].as<std::map<std::string, msgpack::object>>();
    int n_clusters = centers_map["n_clusters"].as<int>();
    int feature_dim = centers_map["feature_dim"].as<int>();
    std::string centers_bytes = centers_map["data"].as<std::string>();

    Eigen::MatrixXf centers(n_clusters, feature_dim);
    std::memcpy(centers.data(), centers_bytes.data(),
                n_clusters * feature_dim * sizeof(float));

    // Deserialize models and metadata...
    return RouterProfile{centers, models, metadata};
}
```

---

## Migration Phases

### Phase 1: Foundation (Week 1-2)

**Goals**: Set up C++ project, export ONNX model, validate embeddings

**Tasks**:
1. Create CMake project structure
2. Set up Conan dependencies (onnxruntime, Eigen, nlohmann_json, msgpack)
3. Export `all-MiniLM-L6-v2` to ONNX format
4. Implement `EmbeddingModel` class with ONNX Runtime
5. Implement tokenization using tokenizers-cpp
6. Validate embeddings match Python (cosine similarity > 0.999)

### Phase 2: Core Logic (Week 3-4)

**Goals**: Implement clustering and scoring logic

**Tasks**:
1. Implement `ClusterEngine` with Eigen
2. Implement `ModelScorer` with heap-based selection
3. Implement profile loading from JSON and binary
4. Implement `Router` main class
5. Unit tests for each component

### Phase 3: FFI Bindings (Week 5-6)

**Goals**: Create bindings for Python

**Tasks**:
1. Implement C API (`adaptive.h`)
2. Create nanobind Python module
3. Integration tests for Python bindings
4. C API available for future Go cgo integration

### Phase 4: Modal Deployment (Week 7-8)

**Goals**: Package for Modal deployment

**Tasks**:
1. Package C++ library for Modal
2. Update Modal image with C++ dependencies (ONNX Runtime + CUDA)
3. Test C++ core loading on Modal T4 GPU
4. Benchmark latency vs Python

### Phase 5: Integration & Validation (Week 9-10)

**Goals**: Validate production readiness

**Tasks**:
1. Shadow deployment (run both C++ and Python, compare results)
2. Load testing (target: 1000 req/s)
3. Accuracy validation (10,000 prompts)
4. Performance benchmarking

### Phase 6: Python Library Migration (Week 11-12)

**Goals**: Update Python router to use C++ core, extend Trainer

**Tasks**:
1. Update `ModelRouter` to use C++ core via nanobind
2. Add `BinaryProfileSaver` to `adaptive_router/savers/`
3. Update `Trainer.train()` to export both JSON and binary formats
4. Update documentation

**Updated Python Usage**:
```python
# Training (exports both formats)
from adaptive_router.core.trainer import Trainer
trainer = Trainer(...)
profile = trainer.train(dataset)
# Exports profile.json and profile.msgpack

# Inference (uses C++ core internally)
from adaptive_router.core.router import ModelRouter
router = ModelRouter.from_local_file("profile.msgpack")
result = router.route("Hello", cost_bias=0.5)
```

---

## Performance Benchmarks

| Metric | Python Baseline | C++ Target (GPU) |
|--------|-----------------|------------------|
| Embedding (single) | 10-20ms GPU | 5-10ms |
| Embedding (batch 32) | 50-100ms GPU | 20-40ms |
| Cluster assignment | <5ms | <1ms |
| Model scoring | <5ms | <0.5ms |
| **Total latency** | **15-30ms** | **7-15ms** |
| Memory footprint | 2-4GB | 200-500MB |
| Cold start | 5-10s | 1-2s |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Embedding mismatch | Validate 10,000+ samples before deployment |
| ONNX export issues | Use official `optimum` library for export |
| Tokenizer differences | Use HuggingFace tokenizers-cpp (same as Python) |
| Memory leaks | Use smart pointers, ASAN in tests |
| Build complexity | Docker for reproducible builds, CI/CD |
| Performance regression | Benchmark before/after, gradual rollout |
| CUDA version mismatch | Pin CUDA 11.8 (same as Modal T4 deployment) |

---

## Deliverables

1. **C++ Core Library** (`libadaptive_core.so` / `.a`) with CUDA support
2. **Python Bindings** (nanobind module compiled into adaptive_router package)
3. **Binary Profile Saver** (Python class in `adaptive_router/savers/binary.py`)
4. **Updated Trainer** (exports both JSON and MessagePack)
5. **Modal Deployment** (Python FastAPI + C++ core)
6. **Documentation** (API unchanged, internal architecture docs)
7. **Validation Scripts** (embedding parity, routing parity)

---

## Critical Files Reference

### Python Files to Modify
- `adaptive_router/core/router.py` - Wrap C++ core instead of Python implementation
- `adaptive_router/core/trainer.py` - Add binary export after training
- `adaptive_router/savers/binary.py` - **New file**: BinaryProfileSaver class

### Python Files to Port to C++
- `adaptive_router/core/router.py:174-204` - Scoring algorithm
- `adaptive_router/core/router.py:611-639` - Lambda/cost normalization
- `adaptive_router/core/cluster_engine.py:324-362` - Cluster assignment
- `adaptive_router/core/cluster_engine.py:191-230` - Embedding extraction
- `adaptive_router/models/storage.py` - Profile data structures
- `adaptive_router/models/routing.py:49-72` - ModelFeatureVector

### Go Files (No Changes Needed)
- `adaptive-proxy/internal/services/model_router/client.go` - HTTP client unchanged

### Modal Files (Minor Updates)
- `main.py` - Update Modal Image with C++ dependencies
