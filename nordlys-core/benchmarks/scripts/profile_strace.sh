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

OUTPUT_FILE="${OUTPUT_DIR}/strace.txt"
SUMMARY_FILE="${OUTPUT_DIR}/strace_summary.txt"

echo "Tracing system calls: $BENCHMARK_FILTER"
echo ""

strace -c -S time \
    -o "$SUMMARY_FILE" \
    "$BENCHMARK_BIN" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=0.1s

echo ""
echo "System call summary:"
cat "$SUMMARY_FILE"

echo ""
echo "Full trace saved: $OUTPUT_FILE"
echo "Summary saved: $SUMMARY_FILE"
