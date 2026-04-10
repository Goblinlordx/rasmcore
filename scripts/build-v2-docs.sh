#!/usr/bin/env bash
# Build SDK and deploy into docs site for development.
#
# Usage: ./scripts/build-v2-docs.sh
#        ./scripts/build-v2-docs.sh --dev    # then starts the dev server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building SDK ==="
"$SCRIPT_DIR/build-sdk.sh"

echo "=== Copying SDK to docs site ==="
rm -rf "$PROJECT_ROOT/docs/site/public/sdk"
mkdir -p "$PROJECT_ROOT/docs/site/public/sdk"
cp -r "$PROJECT_ROOT/sdk/dist/"* "$PROJECT_ROOT/docs/site/public/sdk/"

echo "=== Done ==="
echo "Docs SDK at: $PROJECT_ROOT/docs/site/public/sdk/"

if [ "${1:-}" = "--dev" ]; then
  echo ""
  echo "=== Starting docs dev server ==="
  cd "$PROJECT_ROOT/docs/site"
  exec npm run dev
fi
