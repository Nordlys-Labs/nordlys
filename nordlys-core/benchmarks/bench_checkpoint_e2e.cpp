#include <benchmark/benchmark.h>

#include <nordlys/routing/nordlys.hpp>

#include "bench_utils.hpp"

static void BM_CheckpointLoadJSON_Small(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_small.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
}
BENCHMARK(BM_CheckpointLoadJSON_Small)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoadJSON_Medium(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
}
BENCHMARK(BM_CheckpointLoadJSON_Medium)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoadJSON_Large(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_large.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
}
BENCHMARK(BM_CheckpointLoadJSON_Large)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoadJSON_XL(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_xl.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
}
BENCHMARK(BM_CheckpointLoadJSON_XL)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_Small(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_small.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(nordlys_result);
  }
}
BENCHMARK(BM_RouterInitialization_Small)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_Medium(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(nordlys_result);
  }
}
BENCHMARK(BM_RouterInitialization_Medium)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_Large(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_large.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(nordlys_result);
  }
}
BENCHMARK(BM_RouterInitialization_Large)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_XL(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_xl.json");

  for (auto _ : state) {
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto nordlys_result = Nordlys::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(nordlys_result);
  }
}
BENCHMARK(BM_RouterInitialization_XL)->Unit(benchmark::kMillisecond);

static void BM_CheckpointValidation_Medium(benchmark::State& state) {
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");
  auto checkpoint = NordlysCheckpoint::from_json(path);

  for (auto _ : state) {
    checkpoint.validate();
  }
}
BENCHMARK(BM_CheckpointValidation_Medium)->Unit(benchmark::kMicrosecond);
