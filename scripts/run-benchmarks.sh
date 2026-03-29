#!/usr/bin/env bash
# Run rasmcore performance benchmarks with optional subset selection and report generation.
#
# Usage:
#   ./scripts/run-benchmarks.sh                    # full suite
#   ./scripts/run-benchmarks.sh decoder             # decoder benchmarks only
#   ./scripts/run-benchmarks.sh encoder/jpeg        # JPEG encoder only
#   ./scripts/run-benchmarks.sh filter/bilateral    # bilateral filter only
#   ./scripts/run-benchmarks.sh pipeline            # pipeline chain benchmarks only
#   ./scripts/run-benchmarks.sh --report            # full suite + generate Markdown report
#   ./scripts/run-benchmarks.sh --report decoder    # decoder + report
#   ./scripts/run-benchmarks.sh --list              # list available benchmark groups
#
# Prerequisites:
#   - tests/fixtures/generate.sh must have been run
#   - Reference tools (optional, gracefully skipped if missing):
#     magick, vips, cwebp, dwebp, cjpeg, djpeg, ffmpeg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REPORT=false
FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --report)
            REPORT=true
            shift
            ;;
        --list)
            echo "Available benchmark groups:"
            echo "  decoder          - All decoder benchmarks (JPEG, PNG, WebP)"
            echo "  decoder/jpeg     - JPEG decoder only"
            echo "  decoder/png      - PNG decoder only"
            echo "  decoder/webp     - WebP decoder only"
            echo "  encoder          - All encoder benchmarks"
            echo "  encoder/jpeg     - JPEG encoder only"
            echo "  encoder/png      - PNG encoder only"
            echo "  encoder/webp     - WebP encoder only"
            echo "  filter           - All filter/transform benchmarks"
            echo "  filter/resize    - Resize only"
            echo "  filter/blur      - Blur only"
            echo "  filter/sharpen   - Sharpen only"
            echo "  filter/bilateral - Bilateral filter only"
            echo "  filter/clahe     - CLAHE only"
            echo "  filter/grayscale - Grayscale only"
            echo "  pipeline         - Pipeline chain benchmarks (multi-op)"
            echo ""
            echo "  (no argument)    - Full suite"
            exit 0
            ;;
        *)
            FILTER="$1"
            shift
            ;;
    esac
done

echo "=== rasmcore Performance Benchmarks ==="
echo ""

# Check reference tools
for tool in magick vips cwebp dwebp cjpeg djpeg; do
    if command -v "$tool" &>/dev/null; then
        echo "  [OK] $tool"
    else
        echo "  [SKIP] $tool (not found — those benchmarks will be skipped)"
    fi
done
echo ""

# Ensure fixtures exist
if [ ! -f "$ROOT_DIR/tests/fixtures/generated/inputs/photo_256x256.png" ]; then
    echo "Generating test fixtures..."
    "$ROOT_DIR/tests/fixtures/generate.sh"
fi

# Run benchmarks
cd "$ROOT_DIR"
if [ -n "$FILTER" ]; then
    echo "Running benchmarks matching: $FILTER"
    cargo bench --bench perf -p rasmcore-image -- "$FILTER"
else
    echo "Running full benchmark suite..."
    cargo bench --bench perf -p rasmcore-image
fi

# Generate report if requested
if [ "$REPORT" = true ]; then
    echo ""
    echo "=== Generating Report ==="
    if [ -f "$SCRIPT_DIR/bench-report.py" ]; then
        python3 "$SCRIPT_DIR/bench-report.py" "$ROOT_DIR/target/criterion"
    else
        echo "bench-report.py not found — skipping report generation"
    fi
fi
