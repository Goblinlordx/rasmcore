#!/usr/bin/env bash
# Validation script — runs formatting, linting, and tests.
# Used as pre-merge verification by kiloforge developer workflow.
#
# Usage: ./scripts/validate.sh
#
# Exit codes:
#   0 — all checks pass
#   1 — one or more checks failed

set -euo pipefail

source "$HOME/.cargo/env" 2>/dev/null || true

FAILED=0

echo "=== 1. Format check (rustfmt) ==="
if cargo fmt --check 2>&1; then
  echo "  PASS"
else
  echo "  FAIL — run 'cargo fmt' to fix"
  FAILED=1
fi

echo "=== 1b. License check (cargo-deny) ==="
if command -v cargo-deny &>/dev/null; then
  if cargo deny check licenses 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL — license violation detected (see deny.toml)"
    FAILED=1
  fi
else
  echo "  SKIP — cargo-deny not installed"
fi

echo "=== 2. Lint (clippy) ==="
if cargo clippy --workspace -- -D warnings 2>&1; then
  echo "  PASS"
else
  echo "  FAIL — fix clippy warnings"
  FAILED=1
fi

echo "=== 2b. Build warnings check ==="
BUILD_OUTPUT=$(cargo build --workspace 2>&1)
if echo "$BUILD_OUTPUT" | grep -q "^warning:"; then
  echo "  FAIL — compiler warnings detected:"
  echo "$BUILD_OUTPUT" | grep "^warning:" | head -10
  FAILED=1
else
  echo "  PASS"
fi

echo "=== 3. Domain tests ==="
if cargo test --workspace --lib 2>&1; then
  echo "  PASS"
else
  echo "  FAIL — domain tests failed"
  FAILED=1
fi

echo "=== 4. Parity tests ==="
if [ -d "tests/fixtures/generated" ]; then
  if cargo test --workspace --test parity 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL — parity tests failed"
    FAILED=1
  fi
else
  echo "  SKIP — fixtures not generated (run tests/fixtures/generate.sh)"
fi

echo "=== 5. WASM build ==="
# Generate WIT from filter registrations before WASM build
node scripts/generate-wit.cjs 2>&1 || true
if cargo component build -p rasmcore-image 2>&1; then
  echo "  PASS"
else
  echo "  FAIL — WASM component build failed"
  FAILED=1
fi

echo "=== 6. WASM integration tests ==="
WASM_FILE="target/wasm32-wasip1/debug/rasmcore_image.wasm"
if [ ! -f "$WASM_FILE" ]; then
  WASM_FILE="target/wasm32-wasip1/release/rasmcore_image.wasm"
fi
if [ -f "$WASM_FILE" ] && [ -d "tests/fixtures/generated" ]; then
  if cargo test -p wasm-integration --test wasm_parity 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL — WASM integration tests failed"
    FAILED=1
  fi
else
  echo "  SKIP — requires built .wasm and fixtures"
fi

echo "=== 7. TypeScript SDK tests ==="
if command -v npx &>/dev/null && [ -f "package.json" ] && [ -d "tests/fixtures/generated" ]; then
  # Generate SDK if not already present
  if [ ! -f "sdk/typescript/generated/rasmcore-image.js" ]; then
    if bash scripts/generate-ts-sdk.sh 2>&1; then
      echo "  SDK generated"
    else
      echo "  FAIL — SDK generation failed"
      FAILED=1
    fi
  fi
  if [ -f "sdk/typescript/generated/rasmcore-image.js" ]; then
    if node --test sdk/typescript/tests/ 2>&1; then
      echo "  PASS"
    else
      echo "  FAIL — TypeScript SDK tests failed"
      FAILED=1
    fi
  fi
else
  echo "  SKIP — requires npx, package.json, and fixtures"
fi

echo "=== 8. Demo build ==="
if command -v npm &>/dev/null && [ -d "demo" ] && [ -f "demo/package.json" ]; then
  if (cd demo && npm install --silent 2>/dev/null && npx vite build 2>&1); then
    echo "  PASS"
  else
    echo "  FAIL — demo build failed"
    FAILED=1
  fi
else
  echo "  SKIP — requires npm and demo/ directory"
fi

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "=== ALL CHECKS PASSED ==="
else
  echo "=== SOME CHECKS FAILED ==="
  exit 1
fi
