#!/usr/bin/env bash
# Generate TypeScript SDK from rasmcore WASM components via jco transpile.
#
# Usage: ./scripts/generate-ts-sdk.sh
#
# Prerequisites: npm install, cargo component build -p rasmcore-image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK_DIR="$PROJECT_ROOT/sdk/typescript/generated"

# Find the built WASM component
WASM_FILE=""
for candidate in \
  "$PROJECT_ROOT/target/wasm32-wasip1/debug/rasmcore_image.wasm" \
  "$PROJECT_ROOT/target/wasm32-wasip1/release/rasmcore_image.wasm" \
  "$PROJECT_ROOT/target/wasm32-wasip2/debug/rasmcore_image.wasm" \
  "$PROJECT_ROOT/target/wasm32-wasip2/release/rasmcore_image.wasm"; do
  if [ -f "$candidate" ]; then
    WASM_FILE="$candidate"
    break
  fi
done

if [ -z "$WASM_FILE" ]; then
  echo "ERROR: rasmcore_image.wasm not found."
  echo "Run: cargo component build -p rasmcore-image"
  exit 1
fi

echo "=== Transpiling $WASM_FILE ==="

# Clean previous output
rm -rf "$SDK_DIR"/*.js "$SDK_DIR"/*.d.ts "$SDK_DIR"/*.wasm

# Run jco transpile
npx jco transpile "$WASM_FILE" -o "$SDK_DIR" --name rasmcore-image

echo "=== SDK generated at $SDK_DIR ==="
ls -la "$SDK_DIR"
