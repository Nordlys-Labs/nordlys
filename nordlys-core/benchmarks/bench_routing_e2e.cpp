#include <benchmark/benchmark.h>

#include <mutex>
#include <nordlys/routing/nordlys.hpp>
#include <random>
#include <thread>

#include "bench_utils.hpp"

static NordlysCheckpoint LoadCheckpoint(const std::string& profile_name) {
  return NordlysCheckpoint::from_json(bench_utils::GetFixturePath(profile_name));
}

static void BM_RoutingSingle_Small(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_small.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RoutingSingle_Small)->Unit(benchmark::kMicrosecond);

static void BM_RoutingSingle_Medium(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RoutingSingle_Medium)->Unit(benchmark::kMicrosecond);

static void BM_RoutingSingle_Large(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_large.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RoutingSingle_Large)->Unit(benchmark::kMicrosecond);

static void BM_RoutingSingle_XL(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_xl.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RoutingSingle_XL)->Unit(benchmark::kMicrosecond);



static void BM_RoutingCostBias(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  const float cost_bias = static_cast<float>(state.range(0)) / 100.0f;

  for (auto _ : state) {
    auto result = router.route(embedding.data(), embedding.size(), cost_bias);
    benchmark::DoNotOptimize(result);
  }

  state.SetLabel("lambda=" + std::to_string(cost_bias));
}
BENCHMARK(BM_RoutingCostBias)
    ->Arg(0)
    ->Arg(25)
    ->Arg(50)
    ->Arg(75)
    ->Arg(100)
    ->Unit(benchmark::kMicrosecond);

static void BM_RoutingColdStart_Small(benchmark::State& state) {
  auto embedding = bench_utils::GenerateRandomEmbedding(128);

  for (auto _ : state) {
    auto checkpoint = LoadCheckpoint("checkpoint_small.json");
    auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

    if (!nordlys_result.has_value()) {
      state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
      return;
    }

    auto router = std::move(nordlys_result.value());
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RoutingColdStart_Small)->Unit(benchmark::kMillisecond);

static void BM_RoutingColdStart_Medium(benchmark::State& state) {
  auto embedding = bench_utils::GenerateRandomEmbedding(512);

  for (auto _ : state) {
    auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
    auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

    if (!nordlys_result.has_value()) {
      state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
      return;
    }

    auto router = std::move(nordlys_result.value());
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RoutingColdStart_Medium)->Unit(benchmark::kMillisecond);

static void BM_RoutingConcurrent(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  const int num_threads = state.range(0);
  auto embeddings = bench_utils::GenerateBatchEmbeddings(num_threads, router.get_embedding_dim());

  for (auto _ : state) {
    std::vector<std::thread> threads;
    std::mutex result_mutex;
    std::vector<RouteResult<float>> results;

    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&router, &embeddings, &results, &result_mutex, i]() {
        auto result = router.route(embeddings[i].data(), embeddings[i].size(), 0.5f);
        std::lock_guard<std::mutex> lock(result_mutex);
        results.push_back(std::move(result));
      });
    }

    for (auto& t : threads) {
      t.join();
    }

    benchmark::DoNotOptimize(results);
  }

  state.SetLabel(std::to_string(num_threads) + " threads");
}
BENCHMARK(BM_RoutingConcurrent)
    ->Arg(2)
    ->Arg(4)
    ->Arg(8)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime();

// =============================================================================
// Batch Routing Benchmarks
// =============================================================================

static void BM_RouteBatch(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  const int batch_size = state.range(0);
  const int dim = router.get_embedding_dim();

  std::vector<float> flat_embeddings(batch_size * dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : flat_embeddings) {
    v = dist(gen);
  }

  for (auto _ : state) {
    auto results = router.route_batch(flat_embeddings.data(), batch_size, dim, 0.5f);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_RouteBatch)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Unit(benchmark::kMillisecond);

static void BM_ClusterAssign(benchmark::State& state) {
  const int n_clusters = state.range(0);
  const int dim = 1024;
  
  ClusterEngine<float> engine;
  EmbeddingMatrix<float> centers(n_clusters, dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < centers.rows(); ++i) {
    for (size_t j = 0; j < centers.cols(); ++j) {
      centers(i, j) = dist(gen);
    }
  }
  engine.load_centroids(centers);

  std::vector<float> embedding(dim);
  for (auto& v : embedding) {
    v = dist(gen);
  }

  for (auto _ : state) {
    auto [cluster_id, distance] = engine.assign(embedding.data(), dim);
    benchmark::DoNotOptimize(cluster_id);
    benchmark::DoNotOptimize(distance);
  }

  state.SetLabel(std::to_string(n_clusters) + " clusters");
}
BENCHMARK(BM_ClusterAssign)
    ->Arg(100)
    ->Arg(500)
    ->Arg(1000)
    ->Unit(benchmark::kMicrosecond);
