#!/usr/bin/env bash
# Build self-contained SDK package at sdk/dist/.
#
# Output: sdk/dist/ — a complete npm-publishable package containing:
#   - WASM binary + jco-transpiled JS bindings (wasm/)
#   - Vendored preview2-shim (shim/)
#   - Auto-generated fluent Pipeline class (pipeline.js)
#   - Dynamic RcImage runtime (runtime.js)
#   - Helper libs: gpu-handler, render-target, ml-provider (lib/)
#   - Main entry re-exporting Pipeline + RcImage (rasmcore.js)
#
# Usage: ./scripts/build-sdk.sh
#   --skip-cargo    Skip cargo component build (use existing WASM)
#   --skip-fluent   Skip fluent SDK generation (use existing pipeline.js)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$PROJECT_ROOT/sdk/dist"

WASM_FILE="$PROJECT_ROOT/target/wasm32-wasip1/release/rasmcore_v2_wasm.wasm"
SHIM_SRC="$PROJECT_ROOT/node_modules/@bytecodealliance/preview2-shim/lib/browser"

SKIP_CARGO=false
SKIP_FLUENT=false
for arg in "$@"; do
  case "$arg" in
    --skip-cargo)  SKIP_CARGO=true ;;
    --skip-fluent) SKIP_FLUENT=true ;;
  esac
done

# ─── Step 1: Cargo component build ─────────────────────────────────────────

if [ "$SKIP_CARGO" = false ]; then
  echo "=== [1/6] Building WASM component ==="
  cargo component build -p rasmcore-v2-wasm --release
else
  echo "=== [1/6] Skipping cargo build (--skip-cargo) ==="
fi

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: WASM not found at $WASM_FILE"
  echo "Run without --skip-cargo or: cargo component build -p rasmcore-v2-wasm --release"
  exit 1
fi

# ─── Step 2: jco transpile → sdk/dist/wasm/ ────────────────────────────────

echo "=== [2/6] Transpiling WASM via jco ==="
mkdir -p "$DIST/wasm"

# -M mappings point to ../shim/ (relative to wasm/ subdir → resolves to sdk/dist/shim/)
npx --yes @bytecodealliance/jco transpile "$WASM_FILE" -o "$DIST/wasm" --name rasmcore-v2-image \
  --no-wasi-shim \
  -M "wasi:cli/environment=../shim/cli.js#environment" \
  -M "wasi:cli/exit=../shim/cli.js#exit" \
  -M "wasi:cli/stderr=../shim/cli.js#stderr" \
  -M "wasi:cli/stdin=../shim/cli.js#stdin" \
  -M "wasi:cli/stdout=../shim/cli.js#stdout" \
  -M "wasi:cli/terminal-input=../shim/cli.js#terminalInput" \
  -M "wasi:cli/terminal-output=../shim/cli.js#terminalOutput" \
  -M "wasi:cli/terminal-stderr=../shim/cli.js#terminalStderr" \
  -M "wasi:cli/terminal-stdin=../shim/cli.js#terminalStdin" \
  -M "wasi:cli/terminal-stdout=../shim/cli.js#terminalStdout" \
  -M "wasi:clocks/monotonic-clock=../shim/clocks.js#monotonicClock" \
  -M "wasi:clocks/wall-clock=../shim/clocks.js#wallClock" \
  -M "wasi:filesystem/preopens=../shim/filesystem.js#preopens" \
  -M "wasi:filesystem/types=../shim/filesystem.js#types" \
  -M "wasi:io/error=../shim/io.js#error" \
  -M "wasi:io/streams=../shim/io.js#streams" \
  -M "wasi:random/random=../shim/random.js#random"

# ─── Step 3: Vendor preview2-shim → sdk/dist/shim/ ─────────────────────────

echo "=== [3/6] Vendoring preview2-shim ==="
mkdir -p "$DIST/shim"

if [ -d "$SHIM_SRC" ]; then
  # Copy all browser shim files — modules import from each other internally
  cp "$SHIM_SRC/"*.js "$DIST/shim/"
else
  echo "WARNING: preview2-shim not found at $SHIM_SRC"
  echo "Run: npm install @bytecodealliance/preview2-shim"
  exit 1
fi

# ─── Step 4: Generate fluent Pipeline → sdk/dist/pipeline.ts ───────────────

if [ "$SKIP_FLUENT" = false ]; then
  echo "=== [4/6] Generating fluent Pipeline SDK ==="
  # The generator now outputs directly to sdk/dist/
  SDK_DIST="$DIST" node "$SCRIPT_DIR/generate-v2-fluent-sdk.mjs"
else
  echo "=== [4/6] Skipping fluent SDK generation (--skip-fluent) ==="
fi

# ─── Step 5: Copy helper libs → sdk/dist/lib/ ──────────────────────────────

echo "=== [5/6] Copying helper libs ==="
mkdir -p "$DIST/lib"

LIB_SRC="$PROJECT_ROOT/sdk/v2/lib"
if [ -d "$LIB_SRC" ]; then
  # Copy TypeScript sources — consumers bundle with their own toolchain
  cp "$LIB_SRC/gpu-handler.ts"   "$DIST/lib/" 2>/dev/null || true
  cp "$LIB_SRC/render-target.ts" "$DIST/lib/" 2>/dev/null || true
  cp "$LIB_SRC/ml-provider.ts"   "$DIST/lib/" 2>/dev/null || true

  # Copy shader files
  if [ -d "$LIB_SRC/shaders" ]; then
    mkdir -p "$DIST/lib/shaders"
    cp "$LIB_SRC/shaders/"* "$DIST/lib/shaders/" 2>/dev/null || true
  fi

  # Copy ML subdirectory
  if [ -d "$LIB_SRC/ml" ]; then
    cp -r "$LIB_SRC/ml" "$DIST/lib/"
  fi
fi

# Copy extensions
EXT_SRC="$PROJECT_ROOT/sdk/v2/extensions"
if [ -d "$EXT_SRC" ]; then
  mkdir -p "$DIST/lib/extensions"
  cp "$EXT_SRC/"*.ts "$DIST/lib/extensions/" 2>/dev/null || true
fi

# ─── Step 6: Copy RcImage runtime + main entry ─────────────────────────────

echo "=== [6/6] Assembling main entry ==="

# Copy RcImage runtime from sdks/typescript/src/
RUNTIME_SRC="$PROJECT_ROOT/sdks/typescript/src"
if [ -d "$RUNTIME_SRC" ]; then
  cp "$RUNTIME_SRC/runtime.ts" "$DIST/runtime.ts"
  cp "$RUNTIME_SRC/types.ts"   "$DIST/types.ts"
fi

# Create main entry point
cat > "$DIST/rasmcore.ts" << 'ENTRY_EOF'
/**
 * @rasmcore/image — Professional-grade image processing powered by WebAssembly.
 *
 * Two usage modes:
 *
 * Fluent (typed, auto-generated from WASM registry):
 *   import { Pipeline } from '@rasmcore/image';
 *   const result = Pipeline.open(imageBytes)
 *     .brightness({ amount: 0.5 })
 *     .blur({ radius: 3 })
 *     .writePng();
 *
 * Dynamic (works with any compatible module, no codegen):
 *   import { RcImage } from '@rasmcore/image';
 *   const img = RcImage.load(wasmModule, pngBytes);
 *   const jpeg = img.apply('blur', { radius: 3.0 }).encode('jpeg', { quality: 85 });
 */

export { Pipeline } from './pipeline.js';
export { RcImage } from './runtime.js';
export type {
  FilterManifest,
  OperationMeta,
  ParamMeta,
  WasmModule,
  WasmPipeline,
} from './types.js';
ENTRY_EOF

echo ""
echo "=== SDK built successfully ==="
echo "Output: $DIST/"
ls -la "$DIST/"
echo ""
echo "Subdirectories:"
ls -la "$DIST/wasm/" | head -5
echo "..."
ls "$DIST/shim/"
ls "$DIST/lib/" 2>/dev/null || true
