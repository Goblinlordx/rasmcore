#!/usr/bin/env bash
# Demo build script — compiles WASM component + generates browser SDK
#
# Usage: ./demo/build.sh
# Or:    npm run demo:build

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== 0. Installing demo dependencies ==="
cd "$SCRIPT_DIR"
npm install --silent 2>/dev/null || true
cd "$PROJECT_ROOT"

echo "=== 1a. Generating WIT from build.rs ==="
cd "$PROJECT_ROOT"
# Touch templates to force build.rs to regenerate WIT files
touch wit/image/filters.wit.tmpl wit/image/pipeline.wit.tmpl 2>/dev/null || true

echo "=== 1b. Building WASM component (release) ==="
# build.rs generates filters.wit + pipeline.wit from templates before compilation
cargo component build -p rasmcore-image --release

WASM="$PROJECT_ROOT/target/wasm32-wasip1/release/rasmcore_image.wasm"
if [ ! -f "$WASM" ]; then
    echo "ERROR: WASM binary not found at $WASM"
    exit 1
fi
echo "  WASM: $(du -h "$WASM" | cut -f1) at $WASM"

echo "=== 2. Generating browser SDK via jco transpile ==="
SDK_BUILD_DIR="$PROJECT_ROOT/target/sdk"
mkdir -p "$SDK_BUILD_DIR"
npx @bytecodealliance/jco transpile "$WASM" -o "$SDK_BUILD_DIR/" --name rasmcore-image

echo "=== 3. Copying param manifest ==="
# Find the WASM release manifest (matches the build we just did)
MANIFEST="$PROJECT_ROOT/target/wasm32-wasip1/release/build"
MANIFEST=$(find "$MANIFEST" -name "param-manifest.json" 2>/dev/null | head -1)
if [ -n "$MANIFEST" ]; then
    cp "$MANIFEST" "$SDK_BUILD_DIR/param-manifest.json"
    echo "  Copied: param-manifest.json → target/sdk/"
else
    echo "  WARNING: param-manifest.json not found"
fi

echo "=== 3b. Copying SDK to demo ==="
mkdir -p "$SCRIPT_DIR/sdk"
cp -r "$SDK_BUILD_DIR/"* "$SCRIPT_DIR/sdk/"
echo "  Copied: target/sdk/ → demo/sdk/"

echo "=== 4. Generating fluent SDK (rcimage) ==="
node "$PROJECT_ROOT/scripts/generate-fluent-sdk.cjs"

echo "=== 5. SDK ready ==="
echo "  Output: demo/sdk/"
echo "  Import: import { pipeline, decoder, encoder, filters } from './sdk/rasmcore-image.js'"
echo ""
echo "  Pipeline class: pipeline.ImagePipeline"
echo "  Operations: read, resize, crop, rotate, flip, blur, sharpen,"
echo "              brightness, contrast, grayscale, convolve, median,"
echo "              sobel, canny, composite, writeJpeg/Png/WebP/..."
