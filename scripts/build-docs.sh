#!/usr/bin/env bash
# Build the documentation site (Next.js static export).
#
# Usage: ./scripts/build-docs.sh
# Output: docs/out/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Rendering visual examples ==="
cargo run --bin render_examples -p rasmcore-v2-wasm --release 2>/dev/null

echo "=== Dumping registry ==="
cargo run --bin dump_registry -p rasmcore-v2-wasm 2>/dev/null > /tmp/v2_registry_docs.json

echo "=== Copying example images ==="
mkdir -p "$PROJECT_ROOT/docs/site/public/assets/examples"
cp /tmp/docs-examples/*.png "$PROJECT_ROOT/docs/site/public/assets/examples/" 2>/dev/null || true

echo "=== Copying V2 WASM SDK for live playground ==="
mkdir -p "$PROJECT_ROOT/docs/site/public/sdk/v2"
cp "$PROJECT_ROOT/sdk/typescript/v2-generated/rasmcore-v2-image.js" "$PROJECT_ROOT/docs/site/public/sdk/v2/"
cp "$PROJECT_ROOT/sdk/typescript/v2-generated/rasmcore-v2-image.d.ts" "$PROJECT_ROOT/docs/site/public/sdk/v2/" 2>/dev/null || true
cp "$PROJECT_ROOT/sdk/typescript/v2-generated/"*.wasm "$PROJECT_ROOT/docs/site/public/sdk/v2/"
cp -r "$PROJECT_ROOT/sdk/typescript/v2-generated/interfaces" "$PROJECT_ROOT/docs/site/public/sdk/v2/" 2>/dev/null || true

echo "=== Copying preview2-shim (browser) for import map ==="
SHIM_SRC="$PROJECT_ROOT/docs/site/node_modules/@bytecodealliance/preview2-shim/lib/browser"
SHIM_DST="$PROJECT_ROOT/docs/site/public/sdk/v2/preview2-shim"
if [ -d "$SHIM_SRC" ]; then
  mkdir -p "$SHIM_DST"
  cp "$SHIM_SRC"/*.js "$SHIM_DST/"
fi

echo "=== Building Next.js docs site ==="
cd "$PROJECT_ROOT/docs/site"
npm install --silent
npm run build

echo ""
echo "=== Done ==="
echo "Output: docs/site/out/"
echo "Preview: cd docs/site && npx serve out"
