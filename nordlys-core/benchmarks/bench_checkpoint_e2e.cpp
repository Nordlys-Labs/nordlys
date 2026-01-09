#include <benchmark/benchmark.h>

#include <nordlys_core/nordlys.hpp>

#include "bench_utils.hpp"

static void BM_CheckpointLoadJSON_Small(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_small.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_CheckpointLoadJSON_Small)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoadJSON_Medium(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_CheckpointLoadJSON_Medium)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoadJSON_Large(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_large.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_CheckpointLoadJSON_Large)->Unit(benchmark::kMillisecond);

static void BM_CheckpointLoadJSON_XL(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_xl.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    benchmark::DoNotOptimize(checkpoint);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_CheckpointLoadJSON_XL)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_Small(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_small.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(router_result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RouterInitialization_Small)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_Medium(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(router_result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RouterInitialization_Medium)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_Large(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_large.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(router_result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RouterInitialization_Large)->Unit(benchmark::kMillisecond);

static void BM_RouterInitialization_XL(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_xl.json");

  for (auto _ : state) {
    NORDLYS_ZONE_N("load_checkpoint");
    auto checkpoint = NordlysCheckpoint::from_json(path);
    auto router_result = Nordlys32::from_checkpoint(std::move(checkpoint));
    benchmark::DoNotOptimize(router_result);
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_RouterInitialization_XL)->Unit(benchmark::kMillisecond);

static void BM_CheckpointValidation_Medium(benchmark::State& state) {
  NORDLYS_ZONE;
  const auto path = bench_utils::GetFixturePath("checkpoint_medium.json");
  auto checkpoint = NordlysCheckpoint::from_json(path);

  for (auto _ : state) {
    checkpoint.validate();
  }
  NORDLYS_FRAME_MARK;
}
BENCHMARK(BM_CheckpointValidation_Medium)->Unit(benchmark::kMicrosecond);
