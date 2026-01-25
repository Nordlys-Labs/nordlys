#!/bin/bash
# Simple profiling script for nordlys-core benchmarks
# Usage: bash benchmarks/scripts/profile.sh [benchmark_filter] [duration]

set -e

# Get the nordlys-core directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCHMARKS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NORDLYS_CORE_DIR="$(cd "$BENCHMARKS_DIR/.." && pwd)"
cd "$NORDLYS_CORE_DIR"

# Default values
BENCHMARK_FILTER="${1:-RoutingSingle_Medium}"
PROFILE_DURATION="${2:-10}"
PROFILE_FILE="$BENCHMARKS_DIR/profile.txt"

# Find benchmark executable
BENCH_PATH=$(find build -name "bench_nordlys_core" -type f 2>/dev/null | head -1)

if [ -z "$BENCH_PATH" ]; then
    echo "Benchmark not found. Building..."
    cmake --preset conan-release -DNORDLYS_BUILD_BENCHMARKS=ON
    cmake --build build/Release/build/Release --target bench_nordlys_core
    BENCH_PATH=$(find build -name "bench_nordlys_core" -type f 2>/dev/null | head -1)
fi

if [ -z "$BENCH_PATH" ]; then
    echo "Error: Could not find or build benchmark"
    exit 1
fi

echo "=== Profiling Benchmark ==="
echo "Benchmark: $BENCHMARK_FILTER"
echo "Duration: ${PROFILE_DURATION}s"
echo "Output: $PROFILE_FILE"
echo ""

# Run benchmark in background
echo "Starting benchmark..."
"$BENCH_PATH" --benchmark_filter="$BENCHMARK_FILTER" --benchmark_min_time=1.0 > /dev/null 2>&1 &
BENCH_PID=$!

# Profile for specified duration
echo "Profiling (PID: $BENCH_PID)..."
/usr/bin/sample "$BENCH_PID" "$PROFILE_DURATION" -f "$PROFILE_FILE"

# Wait for benchmark to finish
wait "$BENCH_PID" 2>/dev/null || true

echo ""
echo "=== Profile Analysis ==="
echo ""

# Total samples
TOTAL_SAMPLES=$(grep -E "^[[:space:]]+[0-9]+" "$PROFILE_FILE" | head -1 | awk '{print $1}' || echo "0")
echo "Total samples: $TOTAL_SAMPLES"
echo ""

# Top functions
echo "Top 5 functions:"
grep -A 6 "Sort by top of stack" "$PROFILE_FILE" | head -7
echo ""

# Key metrics
echo "Key metrics:"
echo "  SIMD distance calc: $(grep -c "simsimd_l2sq" "$PROFILE_FILE" || echo "0") references"
echo "  String copies: $(grep -c "__init_copy_ctor_external" "$PROFILE_FILE" || echo "0") (should be 0)"
echo "  Memory allocations: $(grep -c "operator new\|malloc" "$PROFILE_FILE" || echo "0")"
TRACY_COUNT=$(grep -ci tracy "$PROFILE_FILE" 2>/dev/null | head -1 || echo "0")
TRACY_COUNT=$(echo "$TRACY_COUNT" | tr -d '\n' | awk '{print $1}')
if [ -z "$TRACY_COUNT" ] || [ "$TRACY_COUNT" = "0" ]; then
    echo "  Tracy profiler: ✓ Not found (good)"
else
    echo "  Tracy profiler: ⚠ Found $TRACY_COUNT references"
fi

echo ""
echo "Profile saved to: $PROFILE_FILE"
echo "View full details: cat $PROFILE_FILE"
