#!/usr/bin/env bash
# Serve the web-ui for development.
# Usage: ./scripts/web-ui-serve.sh [--sdk-path PATH] [--port PORT]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEB_UI="$ROOT/web-ui"

SDK_PATH="${1:-$ROOT/sdk/typescript/generated}"
PORT="${2:-3000}"

cd "$WEB_UI"

# Ensure dependencies
if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install --silent
fi

# Symlink SDK
if [ -d "$SDK_PATH" ] && [ ! -e "$WEB_UI/sdk" ]; then
  ln -sf "$SDK_PATH" "$WEB_UI/sdk"
  echo "Linked SDK: $SDK_PATH -> web-ui/sdk"
fi

echo "Starting dev server on http://localhost:$PORT"
VITE_SDK_PATH=./sdk npx vite --port "$PORT"
