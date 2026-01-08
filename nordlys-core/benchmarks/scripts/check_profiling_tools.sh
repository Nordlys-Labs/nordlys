#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_tool() {
    local name="$1"
    local cmd="$2"
    if command -v "$cmd" &>/dev/null; then
        printf "${GREEN}✓${NC} %-15s %s\n" "$name" "$(command -v "$cmd")"
        return 0
    else
        printf "${RED}✗${NC} %-15s not found\n" "$name"
        return 1
    fi
}

echo "Checking profiling tools availability..."
echo "========================================="
echo ""

missing=0

echo "Required tools:"
check_tool "perf" "perf" || { missing=$((missing + 1)); }
check_tool "valgrind" "valgrind" || { missing=$((missing + 1)); }
check_tool "heaptrack" "heaptrack" || { missing=$((missing + 1)); }

echo ""
echo "Optional tools:"
check_tool "kcachegrind" "kcachegrind" || true
check_tool "heaptrack_gui" "heaptrack_gui" || true
check_tool "strace" "strace" || true
check_tool "flamegraph" "flamegraph.pl" || true

echo ""
echo "========================================="

if [[ $missing -gt 0 ]]; then
    echo ""
    printf "${YELLOW}Missing $missing required tool(s). Install with:${NC}\n"
    echo ""
    
    if command -v apt &>/dev/null; then
        echo "# Debian/Ubuntu:"
        echo "sudo apt update"
        echo "sudo apt install linux-tools-generic linux-tools-\$(uname -r) valgrind heaptrack kcachegrind strace"
        echo ""
        echo "# FlameGraph (clone to ~/tools or similar):"
        echo "git clone https://github.com/brendangregg/FlameGraph.git ~/tools/FlameGraph"
        echo "export PATH=\"\$PATH:\$HOME/tools/FlameGraph\""
    elif command -v brew &>/dev/null; then
        echo "# macOS (Homebrew):"
        echo "brew install valgrind heaptrack qcachegrind"
        echo ""
        echo "# Note: perf is Linux-only. Use Instruments.app on macOS."
        echo "# FlameGraph:"
        echo "brew install flamegraph"
    elif command -v dnf &>/dev/null; then
        echo "# Fedora/RHEL:"
        echo "sudo dnf install perf valgrind heaptrack kcachegrind strace"
        echo ""
        echo "# FlameGraph:"
        echo "git clone https://github.com/brendangregg/FlameGraph.git ~/tools/FlameGraph"
        echo "export PATH=\"\$PATH:\$HOME/tools/FlameGraph\""
    elif command -v pacman &>/dev/null; then
        echo "# Arch Linux:"
        echo "sudo pacman -S perf valgrind heaptrack kcachegrind strace"
        echo ""
        echo "# FlameGraph (AUR):"
        echo "yay -S flamegraph"
    else
        echo "# Could not detect package manager. Install manually:"
        echo "# - perf (Linux performance counters)"
        echo "# - valgrind (memory/cache analysis)"
        echo "# - heaptrack (heap allocation tracking)"
        echo "# - kcachegrind (callgrind visualizer)"
        echo "# - FlameGraph: https://github.com/brendangregg/FlameGraph"
    fi
    
    echo ""
    printf "${YELLOW}Note: perf requires elevated privileges or adjusted kernel settings.${NC}\n"
    echo "Run with sudo, or set: sudo sysctl kernel.perf_event_paranoid=1"
    exit 1
else
    printf "${GREEN}All required profiling tools are available!${NC}\n"
    exit 0
fi
