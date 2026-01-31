#include <benchmark/benchmark.h>

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

}  // namespace

static void BM_ClusterAssign_CPU(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = state.range(1);

  auto backend = create_backend(CpuDevice{});
  setup_backend(*backend, n_clusters, dim);

  auto query = generate_embedding(dim);
  EmbeddingView view{query.data(), static_cast<size_t>(dim), CpuDevice{}};

  for (auto _ : state) {
    auto [cluster_id, distance] = backend->assign(view);
    benchmark::DoNotOptimize(cluster_id);
    benchmark::DoNotOptimize(distance);
  }

  state.SetLabel(std::to_string(n_clusters) + "c/" + std::to_string(dim) + "d");
}

static void BM_ClusterLoadCentroids_CPU(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = state.range(1);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> centroids(n_clusters * dim);
  for (auto& v : centroids) {
    v = dist(rng);
  }

  for (auto _ : state) {
    auto backend = create_backend(CpuDevice{});
    backend->load_centroids(centroids.data(), n_clusters, dim);
    benchmark::DoNotOptimize(backend);
  }

  state.SetLabel(std::to_string(n_clusters) + "c/" + std::to_string(dim) + "d");
}

static void BM_ClusterBatchAssign_CPU(benchmark::State& state) {
  const int batch_size = state.range(0);
  const int n_clusters = state.range(1);
  const int dim = state.range(2);

  auto backend = create_backend(CpuDevice{});
  setup_backend(*backend, n_clusters, dim);

  std::vector<float> queries(batch_size * dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : queries) {
    v = dist(rng);
  }

  EmbeddingBatchView view{queries.data(), static_cast<size_t>(batch_size),
                          static_cast<size_t>(dim), CpuDevice{}};

  for (auto _ : state) {
    auto results = backend->assign_batch(view);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
  state.SetLabel(std::to_string(batch_size) + "b/" + std::to_string(n_clusters) + "c/" +
                 std::to_string(dim) + "d");
}

static void CPUSingleArgs(benchmark::internal::Benchmark* b) {
  for (int clusters : {10, 50, 100, 500, 1000}) {
    for (int dim : {768, 1024, 1536, 3072, 4096}) {
      b->Args({clusters, dim});
    }
  }
  b->Unit(benchmark::kMicrosecond);
}

static void CPUBatchArgs(benchmark::internal::Benchmark* b) {
  const int clusters = 100;
  const int dim = 1536;
  for (int batch : {1, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    b->Args({batch, clusters, dim});
  }
  b->Unit(benchmark::kMicrosecond);
}

BENCHMARK(BM_ClusterAssign_CPU)->Apply(CPUSingleArgs);
BENCHMARK(BM_ClusterLoadCentroids_CPU)->Apply(CPUSingleArgs);
BENCHMARK(BM_ClusterBatchAssign_CPU)->Apply(CPUBatchArgs);
