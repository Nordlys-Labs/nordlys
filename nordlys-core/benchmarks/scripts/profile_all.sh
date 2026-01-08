#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${BENCH_DIR}/profiling"

BENCHMARK_FILTER="${1:-RoutingSingle_Medium}"

echo "=============================================="
echo "Comprehensive Profiling Suite"
echo "Benchmark: $BENCHMARK_FILTER"
echo "Output: $OUTPUT_DIR"
echo "=============================================="
echo ""

"$SCRIPT_DIR/check_profiling_tools.sh" || {
    echo ""
    echo "Warning: Some tools are missing. Continuing with available tools..."
    echo ""
}

mkdir -p "$OUTPUT_DIR"

run_profile() {
    local name="$1"
    local script="$2"
    
    echo ""
    echo ">>> Running: $name"
    echo "----------------------------------------------"
    
    if [[ -x "$SCRIPT_DIR/$script" ]]; then
        "$SCRIPT_DIR/$script" "$BENCHMARK_FILTER" || {
            echo "Warning: $name failed, continuing..."
        }
    else
        echo "Skipped: $script not found"
    fi
}

run_profile "CPU Performance Counters" "profile_perf.sh"
run_profile "System Call Analysis" "profile_strace.sh"
run_profile "Heap Memory Analysis" "profile_memory.sh"

echo ""
echo "=============================================="
echo "Optional: Long-running profiles"
echo "=============================================="
echo ""
echo "The following profiles take longer to run:"
echo ""
echo "  # Callgrind (cache/branch simulation) - ~5-10 minutes"
echo "  $SCRIPT_DIR/profile_callgrind.sh $BENCHMARK_FILTER"
echo ""
echo "  # Flamegraph (visual CPU profile)"
echo "  $SCRIPT_DIR/profile_flamegraph.sh $BENCHMARK_FILTER"
echo ""
echo "=============================================="
echo "Profiling complete!"
echo "=============================================="
echo ""
echo "Results in: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
