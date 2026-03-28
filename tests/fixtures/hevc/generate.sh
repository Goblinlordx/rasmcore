#!/usr/bin/env bash
# Generate HEVC test fixtures from reference encoder/decoder.
#
# Outputs go to tests/fixtures/hevc/generated/ (gitignored).
# A manifest hash ensures regeneration only when this script changes.
#
# Usage: ./tests/fixtures/hevc/generate.sh [--force]
#
# Requires: x265, ffmpeg
# Optional: dec265 (libde265 CLI) for reference decode verification
#
# Tool versions pinned in TOOLS.toml — warnings on version mismatch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/generated"
HASH_FILE="$OUT_DIR/.manifest_hash"

# ─── Version checks ──────────────────────────────────────────────────────────

check_tool() {
  local tool="$1"
  local expected_version="$2"
  if ! command -v "$tool" &>/dev/null; then
    echo "ERROR: $tool not found. Install it first."
    echo "  See TOOLS.toml for install instructions."
    exit 1
  fi
}

warn_version() {
  local tool="$1"
  local expected="$2"
  local actual="$3"
  if [[ "$actual" != *"$expected"* ]]; then
    echo "WARNING: $tool version mismatch (expected ~$expected, got $actual)"
    echo "  Fixtures may differ. See TOOLS.toml for pinned versions."
  fi
}

check_tool "x265" "4.1"
check_tool "ffmpeg" "7.1"

X265_VERSION=$(x265 --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "unknown")
FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || echo "unknown")

warn_version "x265" "4.1" "$X265_VERSION"
warn_version "ffmpeg" "7.1" "$FFMPEG_VERSION"

# ─── Hash check ──────────────────────────────────────────────────────────────

CURRENT_HASH=$(shasum -a 256 "$0" | awk '{print $1}')

if [[ "${1:-}" != "--force" ]] && [[ -f "$HASH_FILE" ]]; then
  STORED_HASH=$(cat "$HASH_FILE")
  if [[ "$CURRENT_HASH" == "$STORED_HASH" ]]; then
    echo "HEVC fixtures up to date (hash match). Use --force to regenerate."
    exit 0
  fi
fi

echo "=== Generating HEVC test fixtures ==="
mkdir -p "$OUT_DIR"

# ─── Generate raw YUV test images ────────────────────────────────────────────

echo "--- Creating test images (raw YUV 4:2:0) ---"

# 64x64 flat gray (1 CTU — simplest case)
python3 -c "
import sys
w, h = 64, 64
y = bytes([128] * w * h)
u = bytes([128] * (w//2) * (h//2))
v = bytes([128] * (w//2) * (h//2))
sys.stdout.buffer.write(y + u + v)
" > "$OUT_DIR/flat_64x64.yuv"

# 128x128 gradient (4 CTUs — horizontal luma ramp, tests AC coefficients)
python3 -c "
import sys
w, h = 128, 128
y = bytes([int(x / w * 255) for _ in range(h) for x in range(w)])
u = bytes([128] * (w//2) * (h//2))
v = bytes([128] * (w//2) * (h//2))
sys.stdout.buffer.write(y + u + v)
" > "$OUT_DIR/gradient_128x128.yuv"

# 256x256 checkerboard (16 CTUs — complex prediction, alternating 32/224 in 8x8 blocks)
python3 -c "
import sys
w, h = 256, 256
y = []
for row in range(h):
    for col in range(w):
        block = (row // 8 + col // 8) % 2
        y.append(32 if block == 0 else 224)
u = [128] * (w//2) * (h//2)
v = [128] * (w//2) * (h//2)
sys.stdout.buffer.write(bytes(y) + bytes(u) + bytes(v))
" > "$OUT_DIR/checker_256x256.yuv"

echo "  Created: flat_64x64.yuv, gradient_128x128.yuv, checker_256x256.yuv"

# ─── Encode with x265 ───────────────────────────────────────────────────────

encode() {
  local name="$1"
  local width="$2"
  local height="$3"
  local qp="$4"
  local input="$OUT_DIR/${name}.yuv"
  local output="$OUT_DIR/${name}_q${qp}.hevc"

  echo "  Encoding ${name} at QP ${qp} -> $(basename "$output")"
  x265 --input "$input" \
    --input-res "${width}x${height}" \
    --fps 1 \
    --frames 1 \
    --preset ultrafast \
    --keyint 1 \
    --qp "$qp" \
    --no-open-gop \
    --no-cutree \
    --no-scenecut \
    --no-sao \
    --output "$output" \
    2>/dev/null

  # Also create a version with SAO enabled for filter testing
  if [[ "$name" == "checker_256x256" ]]; then
    local sao_output="$OUT_DIR/${name}_q${qp}_sao.hevc"
    echo "  Encoding ${name} at QP ${qp} with SAO -> $(basename "$sao_output")"
    x265 --input "$input" \
      --input-res "${width}x${height}" \
      --fps 1 \
      --frames 1 \
      --preset ultrafast \
      --keyint 1 \
      --qp "$qp" \
      --no-open-gop \
      --no-cutree \
      --no-scenecut \
      --sao \
      --output "$sao_output" \
      2>/dev/null
  fi
}

echo "--- Encoding HEVC I-frames with x265 ---"
encode "flat_64x64"        64   64  22
encode "flat_64x64"        64   64  37
encode "gradient_128x128"  128  128 22
encode "gradient_128x128"  128  128 37
encode "checker_256x256"   256  256 22
encode "checker_256x256"   256  256 37

# ─── Verify with ffmpeg ──────────────────────────────────────────────────────

echo "--- Verifying decode with ffmpeg ---"
FAIL=0
for hevc in "$OUT_DIR"/*.hevc; do
  base=$(basename "$hevc" .hevc)
  decoded="$OUT_DIR/${base}_decoded.yuv"
  if ffmpeg -y -i "$hevc" -f rawvideo -pix_fmt yuv420p "$decoded" 2>/dev/null; then
    echo "  OK: $base"
  else
    echo "  FAIL: $base"
    FAIL=1
  fi
done

if [[ $FAIL -ne 0 ]]; then
  echo "ERROR: Some bitstreams failed to decode!"
  exit 1
fi

# ─── Extract reference data with ffmpeg ──────────────────────────────────────

echo "--- Extracting reference parameter sets and frame data ---"

for hevc in "$OUT_DIR"/*.hevc; do
  [[ "$hevc" == *_decoded* ]] && continue
  base=$(basename "$hevc" .hevc)
  info_file="$OUT_DIR/${base}_info.json"

  # Extract stream info as JSON
  ffprobe -v quiet -print_format json -show_streams "$hevc" > "$info_file" 2>/dev/null || true

  # Extract raw decoded YUV for pixel comparison
  decoded="$OUT_DIR/${base}_decoded.yuv"
  if [[ ! -f "$decoded" ]]; then
    ffmpeg -y -i "$hevc" -f rawvideo -pix_fmt yuv420p "$decoded" 2>/dev/null || true
  fi

  # Extract to RGB for convenience
  rgb_file="$OUT_DIR/${base}_decoded.rgb"
  ffmpeg -y -i "$hevc" -f rawvideo -pix_fmt rgb24 "$rgb_file" 2>/dev/null || true

  echo "  Extracted: ${base} (info, YUV, RGB)"
done

# ─── Extract NAL unit details ────────────────────────────────────────────────

echo "--- Extracting NAL unit structure ---"

for hevc in "$OUT_DIR"/*.hevc; do
  [[ "$hevc" == *_decoded* ]] && continue
  base=$(basename "$hevc" .hevc)
  nal_file="$OUT_DIR/${base}_nals.txt"

  # Use ffprobe to list NAL units
  ffprobe -v quiet -show_packets -show_data "$hevc" > "$nal_file" 2>/dev/null || true

  echo "  NALs: ${base}"
done

# ─── Save manifest hash ─────────────────────────────────────────────────────

echo "$CURRENT_HASH" > "$HASH_FILE"

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "=== HEVC fixtures generated ==="
echo "Location: $OUT_DIR"
echo "Files:"
ls -la "$OUT_DIR"/ | tail -n +2 | awk '{print "  " $NF " (" $5 " bytes)"}'
echo ""
echo "Tool versions used:"
echo "  x265:  $X265_VERSION"
echo "  ffmpeg: $FFMPEG_VERSION"
