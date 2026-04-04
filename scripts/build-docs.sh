#!/usr/bin/env bash
# Build the documentation site from V2 registry + AsciiDoc pages.
#
# Usage: ./scripts/build-docs.sh
# Output: docs/build/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Installing docs dependencies ==="
cd "$PROJECT_ROOT"
npm install --no-save @asciidoctor/core 2>/dev/null || true

echo "=== Building docs site ==="
node "$SCRIPT_DIR/build-docs.mjs"

echo ""
echo "=== Done ==="
echo "Open docs/build/index.html to preview."
echo "Or serve: npx serve docs/build"
