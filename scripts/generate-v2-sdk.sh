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

# Run jco transpile.
# --no-wasi-shim: prevent jco from rewriting WASI imports to bare
#   @bytecodealliance/preview2-shim/* (which browsers can't resolve).
# -M: map each wasi:* interface to the local preview2-shim files with
#   proper relative paths and .js extensions for browser ES modules.
npx --yes @bytecodealliance/jco transpile "$WASM_FILE" -o "$SDK_DIR" --name rasmcore-v2-image \
  --no-wasi-shim \
  -M "wasi:cli/environment=./preview2-shim/cli.js#environment" \
  -M "wasi:cli/exit=./preview2-shim/cli.js#exit" \
  -M "wasi:cli/stderr=./preview2-shim/cli.js#stderr" \
  -M "wasi:cli/stdin=./preview2-shim/cli.js#stdin" \
  -M "wasi:cli/stdout=./preview2-shim/cli.js#stdout" \
  -M "wasi:cli/terminal-input=./preview2-shim/cli.js#terminalInput" \
  -M "wasi:cli/terminal-output=./preview2-shim/cli.js#terminalOutput" \
  -M "wasi:cli/terminal-stderr=./preview2-shim/cli.js#terminalStderr" \
  -M "wasi:cli/terminal-stdin=./preview2-shim/cli.js#terminalStdin" \
  -M "wasi:cli/terminal-stdout=./preview2-shim/cli.js#terminalStdout" \
  -M "wasi:clocks/monotonic-clock=./preview2-shim/clocks.js#monotonicClock" \
  -M "wasi:clocks/wall-clock=./preview2-shim/clocks.js#wallClock" \
  -M "wasi:filesystem/preopens=./preview2-shim/filesystem.js#preopens" \
  -M "wasi:filesystem/types=./preview2-shim/filesystem.js#types" \
  -M "wasi:io/error=./preview2-shim/io.js#error" \
  -M "wasi:io/streams=./preview2-shim/io.js#streams" \
  -M "wasi:random/random=./preview2-shim/random.js#random"

echo ""
echo "=== V2 SDK generated at $SDK_DIR ==="
ls -la "$SDK_DIR"
