#include <benchmark/benchmark.h>

#include <nordlys/checkpoint/checkpoint.hpp>
#include <nordlys/clustering/embedding_view.hpp>
#include <nordlys/common/device.hpp>
#include <nordlys/routing/nordlys.hpp>
#include <random>
#include <thread>
#include <vector>

#include "bench_utils.hpp"

using namespace nordlys::clustering;

namespace {

std::vector<float> generate_embedding(size_t dim, uint32_t seed = 42) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> emb(dim);
  for (auto& v : emb) {
    v = dist(rng);
  }
  return emb;
}

}  // namespace

// =============================================================================
// Checkpoint Loading Benchmarks
// =============================================================================

static void BM_CheckpointLoad_Small(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_small.json");
  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
}
BENCHMARK(BM_CheckpointLoad_Small)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoad_Medium(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");
  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
}
BENCHMARK(BM_CheckpointLoad_Medium)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoad_Large(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_large.json");
  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
}
BENCHMARK(BM_CheckpointLoad_Large)->Unit(benchmark::kMillisecond);

// =============================================================================
// Router Initialization Benchmarks
// =============================================================================

static void BM_RouterInit_Small(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_small.json");
  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto result = Nordlys::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RouterInit_Small)->Unit(benchmark::kMillisecond);

static void BM_RouterInit_Medium(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");
  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto result = Nordlys::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RouterInit_Medium)->Unit(benchmark::kMillisecond);

static void BM_RouterInit_Large(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_large.json");
  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto result = Nordlys::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_RouterInit_Large)->Unit(benchmark::kMillisecond);

// =============================================================================
// Single Routing Benchmarks (CPU backend)
// =============================================================================

static void BM_RouteSingle_Small(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_small.json");
  auto checkpoint = NordlysCheckpoint::from_json(path);
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  if (!result) {
    state.SkipWithError(result.error().c_str());
    return;
  }
  auto& router = result.value();

  auto embedding = generate_embedding(router.get_embedding_dim());
  EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};

  for (auto _ : state) {
    auto route_result = router.route(view);
    benchmark::DoNotOptimize(route_result);
  }
}
BENCHMARK(BM_RouteSingle_Small)->Unit(benchmark::kMicrosecond);

static void BM_RouteSingle_Medium(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");
  auto checkpoint = NordlysCheckpoint::from_json(path);
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  if (!result) {
    state.SkipWithError(result.error().c_str());
    return;
  }
  auto& router = result.value();

  auto embedding = generate_embedding(router.get_embedding_dim());
  EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};

  for (auto _ : state) {
    auto route_result = router.route(view);
    benchmark::DoNotOptimize(route_result);
  }
}
BENCHMARK(BM_RouteSingle_Medium)->Unit(benchmark::kMicrosecond);

static void BM_RouteSingle_Large(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_large.json");
  auto checkpoint = NordlysCheckpoint::from_json(path);
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  if (!result) {
    state.SkipWithError(result.error().c_str());
    return;
  }
  auto& router = result.value();

  auto embedding = generate_embedding(router.get_embedding_dim());
  EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};

  for (auto _ : state) {
    auto route_result = router.route(view);
    benchmark::DoNotOptimize(route_result);
  }
}
BENCHMARK(BM_RouteSingle_Large)->Unit(benchmark::kMicrosecond);

// =============================================================================
// Batch Routing Benchmarks
// =============================================================================

static void BM_RouteBatch(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");
  auto checkpoint = NordlysCheckpoint::from_json(path);
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  if (!result) {
    state.SkipWithError(result.error().c_str());
    return;
  }
  auto& router = result.value();

  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t dim = router.get_embedding_dim();

  std::vector<float> embeddings(batch_size * dim);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : embeddings) {
    v = dist(rng);
  }

  EmbeddingBatchView view{embeddings.data(), batch_size, dim, Device{CpuDevice{}}};

  for (auto _ : state) {
    auto results = router.route_batch(view);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size));
}
BENCHMARK(BM_RouteBatch)
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::kMicrosecond);

// =============================================================================
// Concurrent Routing Benchmarks (Multi-threaded)
// =============================================================================

static void BM_RouteConcurrent(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");
  auto checkpoint = NordlysCheckpoint::from_json(path);
  auto result = Nordlys::from_checkpoint(std::move(checkpoint));
  if (!result) {
    state.SkipWithError(result.error().c_str());
    return;
  }
  auto& router = result.value();

  const int num_threads = state.range(0);
  const size_t dim = router.get_embedding_dim();

  std::vector<std::vector<float>> thread_embeddings(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    thread_embeddings[i] = generate_embedding(dim, 42 + i);
  }

  for (auto _ : state) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back([&router, &thread_embeddings, i]() {
        EmbeddingView view{thread_embeddings[i].data(), thread_embeddings[i].size(), Device{CpuDevice{}}};
        auto route_result = router.route(view);
        benchmark::DoNotOptimize(route_result);
      });
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  state.SetItemsProcessed(state.iterations() * num_threads);
}
BENCHMARK(BM_RouteConcurrent)->Arg(2)->Arg(4)->Arg(8)->Unit(benchmark::kMicrosecond);

// =============================================================================
// Cold Start Benchmark (Load + Init + First Route)
// =============================================================================

static void BM_ColdStart_Medium(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto result = Nordlys::from_checkpoint(std::move(checkpoint));
    if (!result) {
      state.SkipWithError(result.error().c_str());
      return;
    }
    auto& router = result.value();

    auto embedding = generate_embedding(router.get_embedding_dim());
    EmbeddingView view{embedding.data(), embedding.size(), Device{CpuDevice{}}};
    auto route_result = router.route(view);
    benchmark::DoNotOptimize(route_result);
  }
}
BENCHMARK(BM_ColdStart_Medium)->Unit(benchmark::kMillisecond);
