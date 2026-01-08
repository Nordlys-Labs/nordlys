#include <benchmark/benchmark.h>

#include <nordlys_core/cluster.hpp>

#include "bench_utils.hpp"

template <typename Scalar> static void BM_ClusterAssign(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = state.range(1);

  ClusterEngineT<Scalar> engine{ClusterBackendType::Cpu};
  EmbeddingMatrixT<Scalar> centers = EmbeddingMatrixT<Scalar>::Random(n_clusters, dim);
  engine.load_centroids(centers);

  EmbeddingVectorT<Scalar> query = EmbeddingVectorT<Scalar>::Random(dim);

  for (auto _ : state) {
    auto [cluster_id, distance] = engine.assign(query);
    benchmark::DoNotOptimize(cluster_id);
    benchmark::DoNotOptimize(distance);
  }

  state.SetLabel(std::to_string(n_clusters) + "c/" + std::to_string(dim) + "d");
}

static void ClusterArgs(benchmark::internal::Benchmark* b) {
  std::vector<int> cluster_sizes = {10, 50, 100, 500, 1000, 2000};
  std::vector<int> dimensions = {128, 256, 512, 768, 1024, 1536, 3072};

  for (int clusters : cluster_sizes) {
    for (int dim : dimensions) {
      b->Args({clusters, dim});
    }
  }

  b->Unit(benchmark::kMicrosecond);
}

BENCHMARK(BM_ClusterAssign<float>)->Apply(ClusterArgs);
BENCHMARK(BM_ClusterAssign<double>)->Apply(ClusterArgs);

template <typename Scalar> static void BM_ClusterLoadCentroids(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = state.range(1);

  EmbeddingMatrixT<Scalar> centers = EmbeddingMatrixT<Scalar>::Random(n_clusters, dim);

  for (auto _ : state) {
    ClusterEngineT<Scalar> engine{ClusterBackendType::Cpu};
    engine.load_centroids(centers);
    benchmark::DoNotOptimize(engine);
  }

  state.SetLabel(std::to_string(n_clusters) + "c/" + std::to_string(dim) + "d");
}

BENCHMARK(BM_ClusterLoadCentroids<float>)->Apply(ClusterArgs);
BENCHMARK(BM_ClusterLoadCentroids<double>)->Apply(ClusterArgs);

template <typename Scalar> static void BM_ClusterBatchAssign(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = state.range(1);
  const int batch_size = 100;

  ClusterEngineT<Scalar> engine{ClusterBackendType::Cpu};
  EmbeddingMatrixT<Scalar> centers = EmbeddingMatrixT<Scalar>::Random(n_clusters, dim);
  engine.load_centroids(centers);

  std::vector<EmbeddingVectorT<Scalar>> queries;
  queries.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    queries.push_back(EmbeddingVectorT<Scalar>::Random(dim));
  }

  for (auto _ : state) {
    for (const auto& query : queries) {
      auto [cluster_id, distance] = engine.assign(query);
      benchmark::DoNotOptimize(cluster_id);
      benchmark::DoNotOptimize(distance);
    }
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
  state.SetLabel(std::to_string(n_clusters) + "c/" + std::to_string(dim) + "d");
}

static void ClusterBatchArgs(benchmark::internal::Benchmark* b) {
  std::vector<int> cluster_sizes = {10, 100, 1000};
  std::vector<int> dimensions = {128, 512, 1536};

  for (int clusters : cluster_sizes) {
    for (int dim : dimensions) {
      b->Args({clusters, dim});
    }
  }

  b->Unit(benchmark::kMillisecond);
}

BENCHMARK(BM_ClusterBatchAssign<float>)->Apply(ClusterBatchArgs);
BENCHMARK(BM_ClusterBatchAssign<double>)->Apply(ClusterBatchArgs);
