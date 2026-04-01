#!/usr/bin/env bash
# Native benchmark — compare ImageMagick and libvips against GPU PoC results.
# Usage: ./experiments/gpu-poc/bench-native.sh [size]
#
# Requires: ImageMagick (magick), libvips (vips)
# Tests: gaussian blur r=20, spin blur, spherize, bilateral

set -euo pipefail

SIZE=${1:-1024}
RUNS=5
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "═══ Native Benchmark @ ${SIZE}x${SIZE} ═══"
echo ""

# Generate test image
magick -size ${SIZE}x${SIZE} plasma:fractal "$TMPDIR/input.png" 2>/dev/null
echo "Test image: ${SIZE}x${SIZE} PNG"
echo ""

# ─── Helper ──────────────────────────────────────────────────────────────────

bench() {
  local label="$1"
  shift
  local tool="$1"
  shift
  local times=()

  for i in $(seq 1 $RUNS); do
    local t0=$(python3 -c "import time; print(time.perf_counter())")
    eval "$@" > /dev/null 2>&1
    local t1=$(python3 -c "import time; print(time.perf_counter())")
    local ms=$(python3 -c "print(round(($t1 - $t0) * 1000, 1))")
    times+=($ms)
  done

  # Sort and get median
  local sorted=($(printf '%s\n' "${times[@]}" | sort -n))
  local mid=$(( ${#sorted[@]} / 2 ))
  echo "  $label ($tool): ${sorted[$mid]}ms (runs: ${sorted[*]})"
}

# ─── Gaussian Blur r=20 ─────────────────────────────────────────────────────

echo "Gaussian Blur (radius=20, sigma~6.7):"
bench "IM" "magick" "magick '$TMPDIR/input.png' -gaussian-blur 0x6.7 '$TMPDIR/out_im_blur.png'"
bench "vips" "vips" "vips gaussblur '$TMPDIR/input.png' '$TMPDIR/out_vips_blur.png' 6.7"
echo ""

# ─── Spin Blur (radial blur) ────────────────────────────────────────────────

echo "Spin Blur (28.6 degrees / 0.5 rad):"
bench "IM" "magick" "magick '$TMPDIR/input.png' -radial-blur 28.6 '$TMPDIR/out_im_spin.png'"
# vips doesn't have a direct spin/radial blur
echo "  vips: N/A (no radial blur op)"
echo ""

# ─── Spherize ────────────────────────────────────────────────────────────────

echo "Spherize (barrel distortion, strength=0.8):"
bench "IM" "magick" "magick '$TMPDIR/input.png' -distort Barrel '0 0 0.8 0.2' '$TMPDIR/out_im_sphere.png'"
# vips doesn't have a direct spherize
echo "  vips: N/A (no spherize op)"
echo ""

# ─── Bilateral Filter ────────────────────────────────────────────────────────

echo "Bilateral Filter (sigma_s=10, sigma_r=25):"
# IM doesn't have a direct bilateral, use -selective-blur as approximation
bench "IM" "magick" "magick '$TMPDIR/input.png' -selective-blur 5x2+10% '$TMPDIR/out_im_bilateral.png'"
# vips doesn't have bilateral either, skip
echo "  vips: N/A (no bilateral op)"
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────

echo "═══ Reference: GPU PoC Results @ ${SIZE}x${SIZE} ═══"
echo "  GPU (WebGPU compute):  blur=3.8ms  spin=3.0ms  spherize=1.8ms  bilateral=2.8ms"
echo "  WASM (pipeline tiled): blur=140ms  spin=175ms  spherize=265ms"
echo ""
echo "Note: IM/vips times include disk I/O (read PNG + write PNG)."
echo "Subtract ~20-40ms for pure compute comparison."
