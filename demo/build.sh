#!/usr/bin/env bash
# Demo build script — compiles WASM component + generates browser SDK
#
# Usage: ./demo/build.sh
# Or:    npm run demo:build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== 1. Building WASM component (release) ==="
cd "$PROJECT_ROOT"
cargo component build -p rasmcore-image --release

WASM="$PROJECT_ROOT/target/wasm32-wasip1/release/rasmcore_image.wasm"
if [ ! -f "$WASM" ]; then
    echo "ERROR: WASM binary not found at $WASM"
    exit 1
fi
echo "  WASM: $(du -h "$WASM" | cut -f1) at $WASM"

echo "=== 2. Generating browser SDK via jco transpile ==="
mkdir -p "$SCRIPT_DIR/sdk"
npx @bytecodealliance/jco transpile "$WASM" -o "$SCRIPT_DIR/sdk/" --name rasmcore-image

echo "=== 3. SDK ready ==="
echo "  Output: demo/sdk/"
echo "  Import: import { pipeline, decoder, encoder, filters } from './sdk/rasmcore-image.js'"
echo ""
echo "  Pipeline class: pipeline.ImagePipeline"
echo "  Operations: read, resize, crop, rotate, flip, blur, sharpen,"
echo "              brightness, contrast, grayscale, convolve, median,"
echo "              sobel, canny, composite, writeJpeg/Png/WebP/..."
