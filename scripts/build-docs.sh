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

echo "=== Building Next.js docs site ==="
cd "$PROJECT_ROOT/docs/site"
npm install --silent
npm run build

echo ""
echo "=== Done ==="
echo "Output: docs/out/"
echo "Preview: cd docs/site && npx serve ../out"
