#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BENCH_DIR}/../build/Release"
OUTPUT_DIR="${BENCH_DIR}/profiling"
BENCHMARK_BIN="${BUILD_DIR}/benchmarks/bench_nordlys_core"

BENCHMARK_FILTER="${1:-RoutingSingle_Medium}"

if [[ ! -x "$BENCHMARK_BIN" ]]; then
    echo "Error: Benchmark binary not found at $BENCHMARK_BIN"
    echo "Build with: cmake --build --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE="${OUTPUT_DIR}/heaptrack.gz"

echo "Tracking heap allocations: $BENCHMARK_FILTER"
echo ""

heaptrack \
    -o "${OUTPUT_FILE%.gz}" \
    "$BENCHMARK_BIN" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=0.1s

echo ""
echo "Heaptrack output: $OUTPUT_FILE"
echo ""
echo "Analyze with:"
echo "  heaptrack_print $OUTPUT_FILE"
echo "  heaptrack_gui $OUTPUT_FILE"
