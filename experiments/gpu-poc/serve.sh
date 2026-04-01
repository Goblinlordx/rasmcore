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
node -e "
const http = require('http');
const fs = require('fs');
const path = require('path');
const MIME = { '.html':'text/html', '.js':'application/javascript', '.mjs':'application/javascript', '.wgsl':'text/plain', '.wasm':'application/wasm', '.json':'application/json' };
http.createServer((req, res) => {
  const urlPath = new URL(req.url, 'http://localhost').pathname;
  const file = path.join('.', urlPath === '/' ? 'index.html' : decodeURIComponent(urlPath));
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { console.log('404:', file); res.writeHead(404); res.end('Not found: ' + file); return; }
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Content-Type', MIME[path.extname(file)] || 'application/octet-stream');
  fs.createReadStream(file).pipe(res);
}).listen(3000, () => console.log('http://localhost:3000'));
"
