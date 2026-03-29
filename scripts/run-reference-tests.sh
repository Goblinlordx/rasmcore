#!/usr/bin/env bash
# Run ALL reference/parity tests across all crates.
#
# Runs each test suite, collects results, and prints a summary.
# Tests that depend on missing tools or fixtures are skipped gracefully.
#
# Usage: ./scripts/run-reference-tests.sh [--verbose]
#
# Prerequisites: ./scripts/setup-references.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURES_DIR="$ROOT_DIR/tests/fixtures"

VERBOSE=false
[[ "${1:-}" == "--verbose" ]] && VERBOSE=true

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

PASSED=0
FAILED=0
SKIPPED=0
RESULTS=()

run_suite() {
  local name="$1"
  shift
  local cmd=("$@")

  echo -n "  $name ... "

  if $VERBOSE; then
    echo ""
    if "${cmd[@]}" 2>&1; then
      echo -e "  ${GREEN}PASS${NC}"
      RESULTS+=("PASS  $name")
      PASSED=$((PASSED + 1))
    else
      echo -e "  ${RED}FAIL${NC}"
      RESULTS+=("FAIL  $name")
      FAILED=$((FAILED + 1))
    fi
  else
    local output
    if output=$("${cmd[@]}" 2>&1); then
      echo -e "${GREEN}PASS${NC}"
      RESULTS+=("PASS  $name")
      PASSED=$((PASSED + 1))
    else
      echo -e "${RED}FAIL${NC}"
      RESULTS+=("FAIL  $name")
      FAILED=$((FAILED + 1))
      # Show last few lines on failure
      echo "$output" | tail -5 | sed 's/^/    /'
    fi
  fi
}

skip_suite() {
  local name="$1"
  local reason="$2"
  echo -e "  $name ... ${YELLOW}SKIP${NC} ($reason)"
  RESULTS+=("SKIP  $name — $reason")
  SKIPPED=$((SKIPPED + 1))
}

echo "=== Reference Test Suite ==="
echo ""

# ─── 1. Image Reference Audit (ImageMagick) ─────────────────────────────────

echo "--- Image Operations ---"

if command -v magick &>/dev/null; then
  run_suite "reference_audit (ImageMagick)" \
    cargo test -p rasmcore-image --test reference_audit
else
  skip_suite "reference_audit (ImageMagick)" "magick not installed"
fi

# ─── 2. Image Filter Parity (Python) ────────────────────────────────────────

if [[ -f "$FIXTURES_DIR/.venv/bin/python3" ]]; then
  run_suite "filter_reference_parity (Python)" \
    cargo test -p rasmcore-image --test filter_reference_parity
else
  skip_suite "filter_reference_parity (Python)" "venv not set up"
fi

# ─── 3. OpenCV Parity ───────────────────────────────────────────────────────

if [[ -d "$FIXTURES_DIR/opencv" ]] && [[ -f "$FIXTURES_DIR/.venv/bin/python3" ]]; then
  run_suite "opencv_parity" \
    cargo test -p rasmcore-image --test opencv_parity
else
  skip_suite "opencv_parity" "fixtures or venv missing"
fi

echo ""

# ─── 4. JPEG Codec Parity ───────────────────────────────────────────────────

echo "--- Codec Parity ---"

run_suite "rasmcore-jpeg parity" \
  cargo test -p rasmcore-jpeg --test parity

# ─── 5. WebP Codec Parity ───────────────────────────────────────────────────

if command -v cwebp &>/dev/null; then
  run_suite "rasmcore-webp parity" \
    cargo test -p rasmcore-webp --test parity
else
  skip_suite "rasmcore-webp parity" "cwebp not installed"
fi

# ─── 6. HEVC Codec Parity ───────────────────────────────────────────────────

if [[ -d "$FIXTURES_DIR/hevc/generated" ]]; then
  run_suite "rasmcore-hevc parity" \
    cargo test -p rasmcore-hevc --test parity
else
  skip_suite "rasmcore-hevc parity" "HEVC fixtures not generated"
fi

# ─── 7. Multi-Format Reference Parity ───────────────────────────────────────

if [[ -d "$FIXTURES_DIR/generated/reference-parity" ]]; then
  run_suite "codec-parity reference" \
    cargo test -p codec-parity --test reference_parity
else
  skip_suite "codec-parity reference" "reference-parity fixtures not generated"
fi

echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────

TOTAL=$((PASSED + FAILED + SKIPPED))
echo "=== Summary: $PASSED passed, $FAILED failed, $SKIPPED skipped / $TOTAL total ==="
echo ""
for r in "${RESULTS[@]}"; do
  echo "  $r"
done
echo ""

if [[ $FAILED -gt 0 ]]; then
  echo -e "${RED}Some tests failed.${NC} Run with --verbose for details."
  exit 1
elif [[ $SKIPPED -gt 0 ]]; then
  echo -e "${YELLOW}Some tests skipped.${NC} Run ./scripts/setup-references.sh to install dependencies."
  exit 0
else
  echo -e "${GREEN}All reference tests passed.${NC}"
  exit 0
fi
