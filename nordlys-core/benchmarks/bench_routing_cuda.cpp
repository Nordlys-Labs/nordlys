#include <benchmark/benchmark.h>

#ifdef NORDLYS_HAS_CUDA

#  include <nordlys_core/nordlys.hpp>

#  include <random>

#  include "bench_utils.hpp"

static NordlysCheckpoint LoadCheckpoint(const std::string& profile_name) {
  return NordlysCheckpoint::from_json(bench_utils::GetFixturePath(profile_name));
}

static void BM_RoutingGPU_Single_Small(benchmark::State& state) {
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
BENCHMARK(BM_RoutingGPU_Single_Small)->Unit(benchmark::kMicrosecond);

static void BM_RoutingGPU_Single_Medium(benchmark::State& state) {
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
BENCHMARK(BM_RoutingGPU_Single_Medium)->Unit(benchmark::kMicrosecond);

static void BM_RoutingGPU_Single_Large(benchmark::State& state) {
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
BENCHMARK(BM_RoutingGPU_Single_Large)->Unit(benchmark::kMicrosecond);

static void BM_RoutingGPU_Batch(benchmark::State& state) {
  auto checkpoint = LoadCheckpoint("checkpoint_medium.json");
  auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));

  if (!nordlys_result.has_value()) {
    state.SkipWithError(("Failed to create router: " + nordlys_result.error()).c_str());
    return;
  }

  auto router = std::move(nordlys_result.value());
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t dim = router.get_embedding_dim();

  std::vector<float> flat_embeddings(batch_size * dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < flat_embeddings.size(); ++i) {
    flat_embeddings[i] = dist(gen);
  }

  for (auto _ : state) {
    auto results = router.route_batch(flat_embeddings.data(), batch_size, dim, 0.5f);
    benchmark::DoNotOptimize(results);
  }

  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(batch_size));
}
BENCHMARK(BM_RoutingGPU_Batch)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)
    ->Arg(128)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048)->Arg(4096)
    ->Unit(benchmark::kMicrosecond);

static void BM_GPUTransferOverhead_Medium(benchmark::State& state) {
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
BENCHMARK(BM_GPUTransferOverhead_Medium)->Unit(benchmark::kMicrosecond);

#else

static void BM_CUDA_NotAvailable(benchmark::State& state) {
  for (auto _ : state) {
    state.SkipWithMessage("CUDA not available");
  }
}
BENCHMARK(BM_CUDA_NotAvailable);

#endif  // NORDLYS_HAS_CUDA
