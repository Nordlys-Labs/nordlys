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
    echo "Build with: cmake --build --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON -DNORDLYS_BUILD_PROFILE=ON"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

OUTPUT_FILE="${OUTPUT_DIR}/callgrind.out"

echo "Running callgrind on: $BENCHMARK_FILTER"
echo "This may take several minutes (Valgrind adds ~20-50x overhead)..."
echo ""

valgrind \
    --tool=callgrind \
    --callgrind-out-file="$OUTPUT_FILE" \
    --cache-sim=yes \
    --branch-sim=yes \
    --collect-jumps=yes \
    "$BENCHMARK_BIN" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=0.01s

echo ""
echo "Callgrind output: $OUTPUT_FILE"
echo ""
echo "View with: kcachegrind $OUTPUT_FILE"
echo "Or: callgrind_annotate $OUTPUT_FILE | head -100"
