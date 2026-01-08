#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BENCH_DIR}/../build/Release"
OUTPUT_DIR="${BENCH_DIR}/profiling"
BENCHMARK_BIN="${BUILD_DIR}/benchmarks/bench_nordlys_core"

BENCHMARK_FILTER="${1:-RoutingSingle_Medium}"
ITERATIONS="${2:-10}"

if [[ ! -x "$BENCHMARK_BIN" ]]; then
    echo "Error: Benchmark binary not found at $BENCHMARK_BIN"
    echo "Build with: cmake --build --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Profiling: $BENCHMARK_FILTER (${ITERATIONS} iterations)"
echo "Output: ${OUTPUT_DIR}/perf_stat.txt"
echo ""

sudo perf stat \
    -e cycles,instructions,cache-references,cache-misses \
    -e branches,branch-misses \
    -e task-clock,context-switches,cpu-migrations,page-faults \
    --repeat "$ITERATIONS" \
    -o "${OUTPUT_DIR}/perf_stat.txt" \
    "$BENCHMARK_BIN" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=0.1s

echo ""
echo "Results:"
cat "${OUTPUT_DIR}/perf_stat.txt"

# Advanced hardware events (uncomment for deeper analysis on supported CPUs):
# sudo perf stat \
#     -e L1-dcache-loads,L1-dcache-load-misses \
#     -e L1-icache-loads,L1-icache-load-misses \
#     -e LLC-loads,LLC-load-misses \
#     -e dTLB-loads,dTLB-load-misses \
#     -e iTLB-loads,iTLB-load-misses \
#     --repeat "$ITERATIONS" \
#     -o "${OUTPUT_DIR}/perf_stat_advanced.txt" \
#     "$BENCHMARK_BIN" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=0.1s
