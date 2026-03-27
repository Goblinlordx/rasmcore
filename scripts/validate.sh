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

echo "=== 2. Lint (clippy) ==="
if cargo clippy --workspace -- -D warnings 2>&1; then
  echo "  PASS"
else
  echo "  FAIL — fix clippy warnings"
  FAILED=1
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
if cargo component build --workspace 2>&1; then
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

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "=== ALL CHECKS PASSED ==="
else
  echo "=== SOME CHECKS FAILED ==="
  exit 1
fi
