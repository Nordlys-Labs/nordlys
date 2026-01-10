#include <benchmark/benchmark.h>

#ifdef NORDLYS_HAS_CUDA

#  include <nordlys_core/nordlys.hpp>

#  include "bench_utils.hpp"

static NordlysCheckpoint LoadCheckpoint(const std::string& profile_name) {
  return NordlysCheckpoint::from_json(bench_utils::GetFixturePath(profile_name));
}

static void BM_RoutingGPU_Single_Small(benchmark::State& state) {
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
BENCHMARK(BM_RoutingGPU_Single_Small)->Unit(benchmark::kMicrosecond);

static void BM_RoutingGPU_Single_Medium(benchmark::State& state) {
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
BENCHMARK(BM_RoutingGPU_Single_Medium)->Unit(benchmark::kMicrosecond);

static void BM_RoutingGPU_Single_Large(benchmark::State& state) {
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
BENCHMARK(BM_RoutingGPU_Single_Large)->Unit(benchmark::kMicrosecond);

static void BM_RoutingGPU_Batch(benchmark::State& state) {
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
BENCHMARK(BM_RoutingGPU_Batch)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMillisecond);

static void BM_GPUTransferOverhead_Medium(benchmark::State& state) {
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
BENCHMARK(BM_GPUTransferOverhead_Medium)->Unit(benchmark::kMicrosecond);

#else

static void BM_CUDA_NotAvailable(benchmark::State& state) {
  NORDLYS_ZONE;
  for (auto _ : state) {
    state.SkipWithMessage("CUDA not available");
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_CUDA_NotAvailable);

#endif  // NORDLYS_HAS_CUDA
