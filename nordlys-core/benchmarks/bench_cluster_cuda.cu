#ifdef NORDLYS_HAS_CUDA

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <nordlys/clustering/cluster.hpp>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/common/device.hpp>
#include <random>
#include <vector>

using namespace nordlys::clustering;

namespace {

void setup_backend(IClusterBackend& backend, int n_clusters, int dim, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> centroids(n_clusters * dim);
  for (auto& v : centroids) {
    v = dist(rng);
  }
  backend.load_centroids(centroids.data(), n_clusters, dim);
}

std::vector<float> generate_embedding(int dim, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> emb(dim);
  for (auto& v : emb) {
    v = dist(rng);
  }
  return emb;
}

float* allocate_device_embedding(const std::vector<float>& host_data) {
  float* d_ptr = nullptr;
  cudaMalloc(&d_ptr, host_data.size() * sizeof(float));
  cudaMemcpy(d_ptr, host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d_ptr;
}

}  // namespace

// =============================================================================
// Single Assignment - CPU Buffer (includes H2D transfer overhead)
// =============================================================================

static void BM_ClusterAssign_CPUBuffer(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = state.range(1);

  auto backend = create_backend(CudaDevice{0});
  setup_backend(*backend, n_clusters, dim);

  auto query = generate_embedding(dim);
  EmbeddingView view{query.data(), static_cast<size_t>(dim), CpuDevice{}};

  // Warm-up (initializes CUDA graph)
  (void)backend->assign(view);

  for (auto _ : state) {
    auto [cluster_id, distance] = backend->assign(view);
    benchmark::DoNotOptimize(cluster_id);
    benchmark::DoNotOptimize(distance);
  }

  state.SetLabel(std::to_string(n_clusters) + "c/" + std::to_string(dim) + "d");
}

// =============================================================================
// Single Assignment - GPU Buffer (pure compute, PRODUCTION SCENARIO)
// =============================================================================

static void BM_ClusterAssign_GPUBuffer(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = state.range(1);

  auto backend = create_backend(CudaDevice{0});
  setup_backend(*backend, n_clusters, dim);

  auto host_query = generate_embedding(dim);
  float* d_query = allocate_device_embedding(host_query);
  EmbeddingView view{d_query, static_cast<size_t>(dim), CudaDevice{0}};

  // Warm-up
  (void)backend->assign(view);

  for (auto _ : state) {
    auto [cluster_id, distance] = backend->assign(view);
    benchmark::DoNotOptimize(cluster_id);
    benchmark::DoNotOptimize(distance);
  }

  cudaFree(d_query);

  // Mark as production scenario
  state.SetLabel(std::to_string(n_clusters) + "c/" + std::to_string(dim) + "d [PROD]");
}

// =============================================================================
// Batch Assignment - CPU Buffer
// =============================================================================

static void BM_ClusterBatchAssign_CPUBuffer(benchmark::State& state) {
  const int batch_size = state.range(0);
  const int n_clusters = state.range(1);
  const int dim = state.range(2);

  auto backend = create_backend(CudaDevice{0});
  setup_backend(*backend, n_clusters, dim);

  std::vector<float> queries(batch_size * dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : queries) {
    v = dist(rng);
  }

  EmbeddingBatchView view{queries.data(), static_cast<size_t>(batch_size),
                          static_cast<size_t>(dim), CpuDevice{}};

  // Warm-up
  (void)backend->assign_batch(view);

  for (auto _ : state) {
    auto results = backend->assign_batch(view);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
  double throughput = (state.iterations() * batch_size) / (state.iterations() * state.counters["time"] * 1e-6);
  state.counters["throughput_eps"] = benchmark::Counter(throughput, benchmark::Counter::kIsRate);
  
  state.SetLabel(std::to_string(batch_size) + "b/" + std::to_string(n_clusters) + "c/" +
                 std::to_string(dim) + "d");
}

// =============================================================================
// Batch Assignment - GPU Buffer (PRODUCTION SCENARIO)
// =============================================================================

static void BM_ClusterBatchAssign_GPUBuffer(benchmark::State& state) {
  const int batch_size = state.range(0);
  const int n_clusters = state.range(1);
  const int dim = state.range(2);

  auto backend = create_backend(CudaDevice{0});
  setup_backend(*backend, n_clusters, dim);

  // Generate and copy to device
  std::vector<float> host_queries(batch_size * dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : host_queries) {
    v = dist(rng);
  }

  float* d_queries = nullptr;
  cudaMalloc(&d_queries, host_queries.size() * sizeof(float));
  cudaMemcpy(d_queries, host_queries.data(), host_queries.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  EmbeddingBatchView view{d_queries, static_cast<size_t>(batch_size), static_cast<size_t>(dim),
                          CudaDevice{0}};

  // Warm-up
  (void)backend->assign_batch(view);

  for (auto _ : state) {
    auto results = backend->assign_batch(view);
    benchmark::DoNotOptimize(results);
  }

  cudaFree(d_queries);

  state.SetItemsProcessed(state.iterations() * batch_size);
  double throughput = (state.iterations() * batch_size) / (state.iterations() * state.counters["time"] * 1e-6);
  state.counters["throughput_eps"] = benchmark::Counter(throughput, benchmark::Counter::kIsRate);
  
  state.SetLabel(std::to_string(batch_size) + "b/" + std::to_string(n_clusters) + "c/" +
                 std::to_string(dim) + "d [PROD]");
}

// =============================================================================
// Registration
// =============================================================================

static void CUDASingleArgs(benchmark::internal::Benchmark* b) {
  // LLM-focused configurations
  for (int clusters : {10, 50, 100, 500, 1000}) {
    for (int dim : {768, 1024, 1536, 3072, 4096}) {
      b->Args({clusters, dim});
    }
  }
  b->Unit(benchmark::kMicrosecond);
}

static void CUDABatchArgs(benchmark::internal::Benchmark* b) {
  // Fixed cluster/dim, vary batch size (production-focused)
  const int clusters = 100;
  const int dim = 1536;
  for (int batch : {1, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    b->Args({batch, clusters, dim});
  }
  b->Unit(benchmark::kMicrosecond);
}

BENCHMARK(BM_ClusterAssign_CPUBuffer)->Apply(CUDASingleArgs);
BENCHMARK(BM_ClusterAssign_GPUBuffer)->Apply(CUDASingleArgs);
BENCHMARK(BM_ClusterBatchAssign_CPUBuffer)->Apply(CUDABatchArgs);
BENCHMARK(BM_ClusterBatchAssign_GPUBuffer)->Apply(CUDABatchArgs);

#else  // !NORDLYS_HAS_CUDA

#include <benchmark/benchmark.h>

static void BM_CUDA_NotAvailable(benchmark::State& state) {
  for (auto _ : state) {
    state.SkipWithMessage("CUDA not available");
  }
}
BENCHMARK(BM_CUDA_NotAvailable);

#endif  // NORDLYS_HAS_CUDA
