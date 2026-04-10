#!/usr/bin/env bash
# Build V2 SDK and sync into web-ui for development.
#
# Uses build-sdk.sh to produce sdk/dist/, then copies to web-ui/sdk/.
# No sed rewrites — sdk/dist/ has correct paths already.
#
# Usage: ./scripts/build-v2-webui.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEBUI_SDK_DIR="$PROJECT_ROOT/web-ui/sdk"
SDK_DIST="$PROJECT_ROOT/sdk/dist"

echo "=== Building SDK ==="
"$SCRIPT_DIR/build-sdk.sh"

echo "=== Syncing sdk/dist/ → web-ui/sdk/ ==="
rm -rf "$WEBUI_SDK_DIR"
mkdir -p "$WEBUI_SDK_DIR"
cp -r "$SDK_DIST/"* "$WEBUI_SDK_DIR/"

echo "=== Done ==="
echo "Web UI SDK at: $WEBUI_SDK_DIR"
ls -la "$WEBUI_SDK_DIR/"
