#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BENCH_DIR}/../build/Release"
OUTPUT_DIR="${BENCH_DIR}/profiling"
BENCHMARK_BIN="${BUILD_DIR}/benchmarks/bench_nordlys_core"

BENCHMARK_FILTER="${1:-RoutingSingle_Medium}"
FREQUENCY="${2:-99}"

if [[ ! -x "$BENCHMARK_BIN" ]]; then
    echo "Error: Benchmark binary not found at $BENCHMARK_BIN"
    echo "Build with: cmake --build --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON -DNORDLYS_BUILD_PROFILE=ON"
    exit 1
fi

if ! command -v flamegraph.pl &>/dev/null && ! command -v stackcollapse-perf.pl &>/dev/null; then
    echo "Error: FlameGraph scripts not found in PATH"
    echo "Clone: git clone https://github.com/brendangregg/FlameGraph.git ~/tools/FlameGraph"
    echo "Add to PATH: export PATH=\"\$PATH:\$HOME/tools/FlameGraph\""
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PERF_DATA="${OUTPUT_DIR}/perf.data"
FOLDED="${OUTPUT_DIR}/perf.folded"
SVG="${OUTPUT_DIR}/flamegraph.svg"

echo "Recording: $BENCHMARK_FILTER @ ${FREQUENCY}Hz"

sudo perf record \
    -F "$FREQUENCY" \
    -g \
    --call-graph dwarf \
    -o "$PERF_DATA" \
    "$BENCHMARK_BIN" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=1s

echo "Generating flamegraph..."

sudo perf script -i "$PERF_DATA" | stackcollapse-perf.pl > "$FOLDED"
flamegraph.pl "$FOLDED" > "$SVG"

sudo chown "$(id -u):$(id -g)" "$PERF_DATA" "$FOLDED" "$SVG" 2>/dev/null || true

echo ""
echo "Flamegraph saved: $SVG"
echo "Open with: firefox $SVG"
