#!/usr/bin/env bash
# Generate test fixtures and reference outputs using ImageMagick (via Docker).
#
# Outputs go to tests/fixtures/generated/ (gitignored).
# A manifest hash ensures regeneration only when the script changes.
#
# Usage: ./tests/fixtures/generate.sh [--force]
#
# Requires: docker, shasum

set -euo pipefail

# --- Configuration -----------------------------------------------------------
IMAGEMAGICK_IMAGE="dpokidov/imagemagick:7.1.2-12"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/generated"
HASH_FILE="$OUT_DIR/.manifest_hash"

# --- Hash check (skip if already generated from this script version) ---------
CURRENT_HASH=$(shasum -a 256 "$0" | awk '{print $1}')

if [[ "${1:-}" != "--force" ]] && [[ -f "$HASH_FILE" ]]; then
  STORED_HASH=$(cat "$HASH_FILE")
  if [[ "$CURRENT_HASH" == "$STORED_HASH" ]]; then
    echo "Fixtures up to date (hash match). Use --force to regenerate."
    exit 0
  fi
fi

# --- Generate ----------------------------------------------------------------
echo "=== Generating test fixtures (ImageMagick $IMAGEMAGICK_IMAGE) ==="
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/inputs" "$OUT_DIR/reference"

IM="docker run --rm -v $OUT_DIR:/out -w /out --entrypoint magick $IMAGEMAGICK_IMAGE"

# --- Input fixtures ----------------------------------------------------------
echo "  Creating input fixtures..."

# 64x64 gradient (deterministic, good for pixel-exact tests)
$IM -size 64x64 'gradient:red-blue' /out/inputs/gradient_64x64.png
$IM /out/inputs/gradient_64x64.png /out/inputs/gradient_64x64.jpeg
$IM /out/inputs/gradient_64x64.png /out/inputs/gradient_64x64.webp
$IM /out/inputs/gradient_64x64.png /out/inputs/gradient_64x64.gif
$IM /out/inputs/gradient_64x64.png /out/inputs/gradient_64x64.bmp
$IM /out/inputs/gradient_64x64.png /out/inputs/gradient_64x64.tiff
$IM /out/inputs/gradient_64x64.png /out/inputs/gradient_64x64.qoi

# 256x256 fractal plasma (more complex content for quality tests)
# Use a fixed seed for reproducibility
$IM -size 256x256 -seed 42 plasma:fractal /out/inputs/photo_256x256.png
$IM /out/inputs/photo_256x256.png -quality 85 /out/inputs/photo_256x256.jpeg

# --- Reference outputs (ImageMagick as ground truth) -------------------------
echo "  Creating reference outputs..."

# Resize: 64x64 -> 32x16, lanczos
$IM /out/inputs/gradient_64x64.png -resize 32x16! -filter Lanczos /out/reference/resize_lanczos_32x16.png

# Crop: 16x16 at offset (8,8) from 64x64
$IM /out/inputs/gradient_64x64.png -crop 16x16+8+8 +repage /out/reference/crop_16x16_8_8.png

# Rotate
$IM /out/inputs/gradient_64x64.png -rotate 90 /out/reference/rotate_90.png
$IM /out/inputs/gradient_64x64.png -rotate 180 /out/reference/rotate_180.png
$IM /out/inputs/gradient_64x64.png -rotate 270 /out/reference/rotate_270.png

# Flip
$IM /out/inputs/gradient_64x64.png -flop /out/reference/flip_horizontal.png
$IM /out/inputs/gradient_64x64.png -flip /out/reference/flip_vertical.png

# Grayscale
$IM /out/inputs/gradient_64x64.png -colorspace Gray /out/reference/grayscale.png

# Blur (sigma 2.0)
$IM /out/inputs/gradient_64x64.png -blur 0x2 /out/reference/blur_sigma2.png

# --- PNG compression references (for encoder parity tests) ---
echo "  Creating PNG compression references..."
# PNG at compression quality 0 (fastest, least compression)
$IM /out/inputs/photo_256x256.png -quality 0 /out/reference/png_compress_0.png
# PNG at compression quality 30 (low compression)
$IM /out/inputs/photo_256x256.png -quality 30 /out/reference/png_compress_3.png
# PNG at compression quality 60 (default-like compression)
$IM /out/inputs/photo_256x256.png -quality 60 /out/reference/png_compress_6.png
# PNG at compression quality 90 (max compression)
$IM /out/inputs/photo_256x256.png -quality 90 /out/reference/png_compress_9.png

# --- JPEG quality curve references (ImageMagick / libjpeg-turbo) -------------
echo "  Creating JPEG quality curve references..."
for q in 10 30 50 70 85 95; do
  $IM /out/inputs/photo_256x256.png -quality $q /out/reference/jpeg_q${q}.jpeg
done

# --- TIFF compression references (for encoder parity tests) ---
echo "  Creating TIFF compression references..."
# TIFF uncompressed
$IM /out/inputs/photo_256x256.png -compress None /out/reference/tiff_none.tiff
# TIFF LZW
$IM /out/inputs/photo_256x256.png -compress LZW /out/reference/tiff_lzw.tiff
# TIFF Deflate/Zip
$IM /out/inputs/photo_256x256.png -compress Zip /out/reference/tiff_deflate.tiff

# --- Concat reference outputs (ImageMagick append / montage) -----------------
echo "  Creating concat reference outputs..."

# Create two distinct solid-color inputs for concat tests (32x32 red, 32x32 blue)
$IM -size 32x32 'xc:#FF0000' /out/inputs/solid_red_32x32.png
$IM -size 32x32 'xc:#0000FF' /out/inputs/solid_blue_32x32.png
$IM -size 48x24 'xc:#00FF00' /out/inputs/solid_green_48x24.png

# Horizontal append: red + blue side-by-side (64x32)
$IM /out/inputs/solid_red_32x32.png /out/inputs/solid_blue_32x32.png +append /out/reference/concat_h_same_size.png

# Vertical append: red on top, blue on bottom (32x64)
$IM /out/inputs/solid_red_32x32.png /out/inputs/solid_blue_32x32.png -append /out/reference/concat_v_same_size.png

# Horizontal with different heights: red 32x32 + green 48x24 (gray bg, centered)
$IM /out/inputs/solid_red_32x32.png /out/inputs/solid_green_48x24.png -background '#808080' -gravity Center +append /out/reference/concat_h_diff_height.png

# Vertical with different widths: red 32x32 + green 48x24 (gray bg, centered)
$IM /out/inputs/solid_red_32x32.png /out/inputs/solid_green_48x24.png -background '#808080' -gravity Center -append /out/reference/concat_v_diff_width.png

# --- Verify reproducibility (hash all outputs) -------------------------------
echo "  Verifying output hashes..."
OUTPUT_HASH=$(cd "$OUT_DIR" && find . \( -name '*.png' -o -name '*.jpeg' -o -name '*.webp' -o -name '*.gif' -o -name '*.bmp' -o -name '*.tiff' -o -name '*.qoi' \) | sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}')
echo "$OUTPUT_HASH" > "$OUT_DIR/.output_hash"

# --- Store script hash for cache invalidation --------------------------------
echo "$CURRENT_HASH" > "$HASH_FILE"

echo "=== Done. Fixtures in $OUT_DIR ==="
echo "  Inputs:     $(ls "$OUT_DIR/inputs/" | wc -l | tr -d ' ') files"
echo "  References: $(ls "$OUT_DIR/reference/" | wc -l | tr -d ' ') files"
echo "  Output hash: $OUTPUT_HASH"
