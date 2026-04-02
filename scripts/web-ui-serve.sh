#!/usr/bin/env bash
# Serve the web-ui for development.
# Usage: ./scripts/web-ui-serve.sh [SDK_PATH] [PORT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_UI="$ROOT/web-ui"

SDK_PATH="${1:-$ROOT/sdk/typescript/generated}"
PORT="${2:-3000}"

# Verify SDK exists — rebuild if missing
if [ ! -f "$SDK_PATH/rasmcore-image.js" ]; then
  echo "SDK not found — building WASM component and generating SDK..."
  cargo component build -p rasmcore-image
  "$ROOT/scripts/generate-ts-sdk.sh"
fi

# Copy SDK files into web-ui/sdk/ (workers import from ../sdk/)
echo "Syncing SDK into web-ui/sdk/..."
mkdir -p "$WEB_UI/sdk"
rsync -a --delete --exclude='.gitignore' "$SDK_PATH/" "$WEB_UI/sdk/"

cd "$WEB_UI"

# Ensure dependencies
if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install --silent
fi

echo "Starting dev server on http://localhost:$PORT"
VITE_SDK_PATH=./sdk npx vite --port "$PORT"
