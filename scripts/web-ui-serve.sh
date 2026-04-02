#!/usr/bin/env bash
# Serve the web-ui for development.
# Usage: ./scripts/web-ui-serve.sh [SDK_PATH] [PORT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_UI="$ROOT/web-ui"

SDK_PATH="${1:-$ROOT/sdk/typescript/generated}"
SDK_JS="$SDK_PATH/rasmcore-image.js"
WASM_FILE="$ROOT/target/wasm32-wasip1/debug/rasmcore_image.wasm"
PORT="${2:-3000}"

# Build WASM if missing
if [ ! -f "$WASM_FILE" ]; then
  echo "Building WASM component..."
  cargo component build -p rasmcore-image
fi

# Generate SDK if missing or stale (older than WASM)
if [ ! -f "$SDK_JS" ] || [ "$WASM_FILE" -nt "$SDK_JS" ]; then
  echo "Generating TypeScript SDK..."
  "$ROOT/scripts/generate-ts-sdk.sh"
fi

# Copy SDK files into web-ui/sdk/ (workers import from ../sdk/)
echo "Syncing SDK into web-ui/sdk/..."
mkdir -p "$WEB_UI/sdk"
cp -R "$SDK_PATH/"* "$WEB_UI/sdk/" 2>/dev/null || true

cd "$WEB_UI"

# Ensure dependencies
if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install --silent
fi

echo "Starting dev server on http://localhost:$PORT"
VITE_SDK_PATH=./sdk npx vite --port "$PORT"
