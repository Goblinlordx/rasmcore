#!/usr/bin/env bash
# Serve the GPU PoC benchmark with required COOP/COEP headers.
# Usage: ./experiments/gpu-poc/serve.sh
#
# Requires: npx (comes with npm)
# Open: http://localhost:3000

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Copy SDK so the benchmark can import it
if [ -d "$PROJECT_ROOT/web-ui/sdk" ] && [ "$(ls -A "$PROJECT_ROOT/web-ui/sdk" 2>/dev/null)" ]; then
  mkdir -p "$SCRIPT_DIR/sdk"
  cp -r "$PROJECT_ROOT/web-ui/sdk/"* "$SCRIPT_DIR/sdk/" 2>/dev/null || true
  echo "SDK copied from web-ui/sdk/"
elif [ -d "$PROJECT_ROOT/target/sdk" ]; then
  mkdir -p "$SCRIPT_DIR/sdk"
  cp -r "$PROJECT_ROOT/target/sdk/"* "$SCRIPT_DIR/sdk/" 2>/dev/null || true
  echo "SDK copied from target/sdk/"
else
  echo "WARNING: No SDK found. Run demo/build.sh first."
  echo "WASM baseline will be unavailable (GPU benchmarks still work)."
fi

echo "Serving GPU PoC at http://localhost:3000"
echo "Press Ctrl+C to stop."
echo ""

cd "$SCRIPT_DIR"
npx serve -l 3000 \
  --cors \
  -H "Cross-Origin-Opener-Policy: same-origin" \
  -H "Cross-Origin-Embedder-Policy: require-corp"
