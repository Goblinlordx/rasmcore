#!/usr/bin/env bash
# Generate TypeScript SDK from V2 WASM component via jco transpile.
#
# Usage: ./scripts/generate-v2-sdk.sh
#
# Prerequisites: npm install, cargo component build -p rasmcore-v2-wasm --release

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SDK_DIR="$PROJECT_ROOT/sdk/typescript/v2-generated"

WASM_FILE="$PROJECT_ROOT/target/wasm32-wasip1/release/rasmcore_v2_wasm.wasm"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: V2 WASM not found at $WASM_FILE"
  echo "Run: cargo component build -p rasmcore-v2-wasm --release"
  exit 1
fi

echo "=== Transpiling V2 WASM: $WASM_FILE ==="

mkdir -p "$SDK_DIR"

# Run jco transpile
npx --yes @bytecodealliance/jco transpile "$WASM_FILE" -o "$SDK_DIR" --name rasmcore-v2-image

echo ""
echo "=== V2 SDK generated at $SDK_DIR ==="
ls -la "$SDK_DIR"
