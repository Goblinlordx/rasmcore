#!/usr/bin/env bash
# Build V2 WASM + SDK and sync into web-ui for development.
#
# Usage: ./scripts/build-v2-webui.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEBUI_SDK_DIR="$PROJECT_ROOT/web-ui/sdk/v2"

echo "=== Building V2 WASM component ==="
cargo component build -p rasmcore-v2-wasm --release

echo "=== Generating V2 raw SDK (jco transpile) ==="
"$SCRIPT_DIR/generate-v2-sdk.sh"

echo "=== Generating V2 fluent SDK ==="
node "$SCRIPT_DIR/generate-v2-fluent-sdk.mjs"

echo "=== Syncing into web-ui/sdk/v2/ ==="
mkdir -p "$WEBUI_SDK_DIR"
cp -r "$PROJECT_ROOT/sdk/typescript/v2-generated/"* "$WEBUI_SDK_DIR/"
mkdir -p "$WEBUI_SDK_DIR/fluent"
# Fix import path: in the source tree it's ../v2-generated/interfaces/
# but in web-ui/sdk/v2/fluent/ the interfaces dir is at ../interfaces/
sed 's|../v2-generated/interfaces/|../interfaces/|g' \
  "$PROJECT_ROOT/sdk/typescript/v2-fluent/index.ts" > "$WEBUI_SDK_DIR/fluent/index.ts"

# Copy SDK library files (render-target, gpu-handler, shaders)
if [ -d "$PROJECT_ROOT/sdk/v2/lib" ]; then
  cp -r "$PROJECT_ROOT/sdk/v2/lib" "$WEBUI_SDK_DIR/lib"
  echo "  Synced sdk/v2/lib/"
fi

echo "=== Done ==="
echo "Web UI SDK at: $WEBUI_SDK_DIR"
ls -la "$WEBUI_SDK_DIR/"
