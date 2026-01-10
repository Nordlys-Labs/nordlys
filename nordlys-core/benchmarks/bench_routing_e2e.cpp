#include <benchmark/benchmark.h>

#include <mutex>
#include <nordlys_core/nordlys.hpp>
#include <thread>

#include "bench_utils.hpp"

static NordlysCheckpoint LoadCheckpoint(const std::string& profile_name) {
  return NordlysCheckpoint::from_json(bench_utils::GetFixturePath(profile_name));
}

static void BM_RoutingSingle_Small(benchmark::State& state) {
  NORDLYS_ZONE;
  auto checkpoint = LoadCheckpoint("checkpoint_small.json");
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

  if (!router_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
    return;
  }

  auto router = std::move(router_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    NORDLYS_ZONE_N("route_iteration");
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RoutingSingle_Small)->Unit(benchmark::kMicrosecond);

static void BM_RoutingSingle_Medium(benchmark::State& state) {
  NORDLYS_ZONE;
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

  if (!router_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
    return;
  }

  auto router = std::move(router_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    NORDLYS_ZONE_N("route_iteration");
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RoutingSingle_Medium)->Unit(benchmark::kMicrosecond);

static void BM_RoutingSingle_Large(benchmark::State& state) {
  NORDLYS_ZONE;
  auto checkpoint = LoadCheckpoint("checkpoint_large.json");
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

  if (!router_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
    return;
  }

  auto router = std::move(router_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    NORDLYS_ZONE_N("route_iteration");
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RoutingSingle_Large)->Unit(benchmark::kMicrosecond);

static void BM_RoutingSingle_XL(benchmark::State& state) {
  NORDLYS_ZONE;
  auto checkpoint = LoadCheckpoint("checkpoint_xl.json");
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

  if (!router_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
    return;
  }

  auto router = std::move(router_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  for (auto _ : state) {
    NORDLYS_ZONE_N("route_iteration");
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RoutingSingle_XL)->Unit(benchmark::kMicrosecond);

static void BM_RoutingBatch(benchmark::State& state) {
  NORDLYS_ZONE;
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

  if (!router_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
    return;
  }

  auto router = std::move(router_result.value());
  const int batch_size = state.range(0);
  auto embeddings = bench_utils::GenerateBatchEmbeddings(batch_size, router.get_embedding_dim());

  for (auto _ : state) {
    for (const auto& emb : embeddings) {
      auto result = router.route(emb.data(), emb.size(), 0.5f);
      benchmark::DoNotOptimize(result);
    }
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_RoutingBatch)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMillisecond);

static void BM_RoutingCostBias(benchmark::State& state) {
  NORDLYS_ZONE;
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

  if (!router_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
    return;
  }

  auto router = std::move(router_result.value());
  auto embedding = bench_utils::GenerateRandomEmbedding(router.get_embedding_dim());

  const float cost_bias = static_cast<float>(state.range(0)) / 100.0f;

  for (auto _ : state) {
    NORDLYS_ZONE_N("route_iteration");
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
  NORDLYS_ZONE;
  auto embedding = bench_utils::GenerateRandomEmbedding(128);

  for (auto _ : state) {
    auto checkpoint = LoadCheckpoint("checkpoint_small.json");
    auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

    if (!router_result.has_value()) {
      state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
      return;
    }

    auto router = std::move(router_result.value());
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RoutingColdStart_Small)->Unit(benchmark::kMillisecond);

static void BM_RoutingColdStart_Medium(benchmark::State& state) {
  NORDLYS_ZONE;
  auto embedding = bench_utils::GenerateRandomEmbedding(512);

  for (auto _ : state) {
    auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
    auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

    if (!router_result.has_value()) {
      state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
      return;
    }

    auto router = std::move(router_result.value());
    auto result = router.route(embedding.data(), embedding.size(), 0.5f);
    benchmark::DoNotOptimize(result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RoutingColdStart_Medium)->Unit(benchmark::kMillisecond);

static void BM_RoutingConcurrent(benchmark::State& state) {
  NORDLYS_ZONE;
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));

  if (!router_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + router_result.error()).c_str());
    return;
  }

  auto router = std::move(router_result.value());
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
