#!/usr/bin/env bash
# Build V2 WASM + SDK and deploy into docs site for development.
#
# Usage: ./scripts/build-v2-docs.sh
#        ./scripts/build-v2-docs.sh --dev    # then starts the dev server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_SDK_DIR="$PROJECT_ROOT/docs/site/public/sdk/v2"

echo "=== Building V2 WASM component (release) ==="
cargo component build -p rasmcore-v2-wasm --release

echo "=== Generating V2 SDK (jco transpile) ==="
"$SCRIPT_DIR/generate-v2-sdk.sh"

echo "=== Generating V2 fluent SDK ==="
node "$SCRIPT_DIR/generate-v2-fluent-sdk.mjs"

echo "=== Copying SDK to docs site ==="
mkdir -p "$DOCS_SDK_DIR"
cp -r "$PROJECT_ROOT/sdk/typescript/v2-generated/"* "$DOCS_SDK_DIR/"

echo "=== Done ==="
echo "Docs SDK at: $DOCS_SDK_DIR"

if [ "${1:-}" = "--dev" ]; then
  echo ""
  echo "=== Starting docs dev server ==="
  cd "$PROJECT_ROOT/docs/site"
  exec npm run dev
fi
