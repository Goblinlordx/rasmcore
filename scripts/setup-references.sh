#!/usr/bin/env bash
# Setup all external reference tools and generate test fixtures.
#
# Installs: ImageMagick, libwebp (cwebp), libjpeg-turbo (cjpeg/djpeg),
#           ffmpeg, x265, Python venv (numpy, Pillow, OpenCV).
# Generates: all test fixtures from existing generation scripts.
#
# Usage: ./scripts/setup-references.sh [--skip-install] [--force]
#
# Options:
#   --skip-install  Skip tool installation, only generate fixtures
#   --force         Force regeneration of all fixtures
#
# Idempotent: safe to run multiple times.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FIXTURES_DIR="$ROOT_DIR/tests/fixtures"
VENV_DIR="$FIXTURES_DIR/.venv"

SKIP_INSTALL=false
FORCE=""
for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=true ;;
    --force) FORCE="--force" ;;
  esac
done

# ─── Colors ─────────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

# ─── OS Detection ───────────────────────────────────────────────────────────

detect_os() {
  case "$(uname -s)" in
    Darwin*) echo "macos" ;;
    Linux*)  echo "linux" ;;
    *)       echo "unknown" ;;
  esac
}

OS=$(detect_os)

# ─── Tool Installation ──────────────────────────────────────────────────────

install_tool() {
  local name="$1"
  local brew_pkg="$2"
  local apt_pkg="${3:-$brew_pkg}"

  if command -v "$name" &>/dev/null; then
    ok "$name already installed ($(command -v "$name"))"
    return 0
  fi

  if $SKIP_INSTALL; then
    warn "$name not found (--skip-install)"
    return 1
  fi

  echo "  Installing $name..."
  if [[ "$OS" == "macos" ]]; then
    if command -v brew &>/dev/null; then
      brew install "$brew_pkg" 2>/dev/null && ok "$name installed" && return 0
    fi
    fail "$name: brew not available"
    return 1
  elif [[ "$OS" == "linux" ]]; then
    if command -v apt-get &>/dev/null; then
      sudo apt-get install -y "$apt_pkg" 2>/dev/null && ok "$name installed" && return 0
    fi
    fail "$name: apt-get not available"
    return 1
  fi
  fail "$name: unsupported OS"
  return 1
}

echo "=== Reference Test Setup ==="
echo "OS: $OS"
echo ""

# ─── 1. Install External Tools ──────────────────────────────────────────────

echo "--- 1. External Tools ---"

TOOLS_OK=0
TOOLS_MISSING=0

check_or_install() {
  local name="$1"
  local brew="$2"
  local apt="${3:-$2}"
  if install_tool "$name" "$brew" "$apt"; then
    TOOLS_OK=$((TOOLS_OK + 1))
  else
    TOOLS_MISSING=$((TOOLS_MISSING + 1))
  fi
}

check_or_install "magick"   "imagemagick" "imagemagick"
check_or_install "cwebp"    "webp"        "webp"
check_or_install "cjpeg"    "mozjpeg"     "libjpeg-turbo-progs"
check_or_install "ffmpeg"   "ffmpeg"      "ffmpeg"
check_or_install "ffprobe"  "ffmpeg"      "ffmpeg"
check_or_install "x265"     "x265"        "x265"

# vips is optional
if command -v vips &>/dev/null; then
  ok "vips available (optional)"
else
  warn "vips not found (optional — some comparison benchmarks will skip)"
fi

echo ""
echo "  Tools: $TOOLS_OK available, $TOOLS_MISSING missing"
echo ""

# ─── 2. Python Virtual Environment ──────────────────────────────────────────

echo "--- 2. Python Virtual Environment ---"

if [[ -f "$VENV_DIR/bin/python3" ]]; then
  ok "venv exists at $VENV_DIR"
else
  echo "  Creating venv..."
  python3 -m venv "$VENV_DIR"
  ok "venv created"
fi

# Install/upgrade packages
echo "  Installing Python packages..."
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet numpy Pillow opencv-python-headless
ok "numpy, Pillow, opencv-python-headless installed"

# Verify
"$VENV_DIR/bin/python3" -c "import numpy, PIL, cv2; print(f'  numpy={numpy.__version__}, Pillow={PIL.__version__}, OpenCV={cv2.__version__}')"
echo ""

# ─── 3. Generate Fixtures ───────────────────────────────────────────────────

echo "--- 3. Generate Fixtures ---"

# Main fixtures (Docker-based)
if [[ -f "$FIXTURES_DIR/generate.sh" ]]; then
  echo "  Running generate.sh..."
  if bash "$FIXTURES_DIR/generate.sh" $FORCE 2>&1 | tail -1; then
    ok "Main fixtures generated"
  else
    warn "Main fixture generation failed (Docker may be unavailable)"
  fi
else
  warn "generate.sh not found"
fi

# Reference parity fixtures (local tools)
if [[ -f "$FIXTURES_DIR/generate-reference.sh" ]]; then
  echo "  Running generate-reference.sh..."
  if bash "$FIXTURES_DIR/generate-reference.sh" $FORCE 2>&1 | tail -1; then
    ok "Reference parity fixtures generated"
  else
    warn "Reference parity fixture generation failed"
  fi
else
  warn "generate-reference.sh not found"
fi

# HEVC fixtures
if [[ -f "$FIXTURES_DIR/hevc/generate.sh" ]]; then
  if command -v x265 &>/dev/null && command -v ffmpeg &>/dev/null; then
    echo "  Running hevc/generate.sh..."
    if bash "$FIXTURES_DIR/hevc/generate.sh" $FORCE 2>&1 | tail -1; then
      ok "HEVC fixtures generated"
    else
      warn "HEVC fixture generation failed"
    fi
  else
    warn "HEVC fixtures skipped (x265 or ffmpeg not available)"
  fi
else
  warn "hevc/generate.sh not found"
fi

echo ""

# ─── 4. Summary ─────────────────────────────────────────────────────────────

echo "=== Setup Complete ==="
echo ""
echo "Available test suites:"

check_suite() {
  local name="$1"
  local required="$2"
  if command -v "$required" &>/dev/null; then
    ok "$name"
  else
    fail "$name (missing: $required)"
  fi
}

check_suite "Image reference audit (ImageMagick)"    "magick"
check_suite "Image filter parity (Python)"           "$VENV_DIR/bin/python3"
check_suite "OpenCV parity (Python+OpenCV)"          "$VENV_DIR/bin/python3"
check_suite "JPEG codec parity (cjpeg)"              "cjpeg"
check_suite "WebP codec parity (cwebp)"              "cwebp"

if [[ -d "$FIXTURES_DIR/hevc/generated" ]]; then
  ok "HEVC codec parity (fixtures present)"
else
  warn "HEVC codec parity (run hevc/generate.sh with x265+ffmpeg)"
fi

if [[ -d "$FIXTURES_DIR/generated/reference-parity" ]]; then
  ok "Multi-format parity (fixtures present)"
else
  warn "Multi-format parity (run generate-reference.sh)"
fi

echo ""
echo "Run all reference tests:  ./scripts/run-reference-tests.sh"
