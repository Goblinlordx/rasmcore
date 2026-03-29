# Reference Validation

Every codec, filter, and transform in rasmcore is validated against an
authoritative reference implementation. This document is the single source
of truth for what was validated, against what, at what version, and to
what precision.

## Validation Principle

**Exact match is the default.** If our output differs from the reference by
even one byte, we either fix our implementation or document exactly why the
difference exists (e.g., f32 cross-platform rounding).

No tests skip. If a reference tool is unavailable, the test **fails** with
setup instructions.

---

## Codec Validation

### VP8 / WebP Encoder (rasmcore-webp)

| Aspect | Reference | Version | Result |
|--------|-----------|---------|--------|
| Bitstream conformance | dwebp (libwebp) | 1.6.0 | Decodable at all sizes 16-4096 |
| Bitstream conformance | image-webp (Rust) | 0.2.4 | Decodable at all sizes 16-4096 |
| BoolWriter arithmetic | libvpx `vp8_encode_bool` | N/A | Roundtrip verified with BoolReader |
| SATD (Hadamard) | libvpx `vp8_short_walsh4x4_c` | N/A | 6 golden vectors, exact match |
| Coefficient probs | libvpx `kDefaultCoefProbs` | BSD-3 | Byte-identical 1056-entry table |
| Update probs | libvpx `vp8_coef_update_probs` | BSD-3 | Byte-identical 1056-entry table |
| Y mode tree | RFC 6386 Section 11.2 | N/A | Verified against image-webp decoder |
| Token tree | RFC 6386 Section 13 | N/A | Verified against image-webp decoder |
| DC/AC quant tables | RFC 6386 Section 14.1 | N/A | Identical to image-webp DC_QUANT/AC_QUANT |

**PSNR at 256x256 gradient:**
- q75: 27.6 dB (vs cwebp 44.9 dB — gap from missing B_PRED + segmentation)
- q100: 46.4 dB (vs cwebp 47.4 dB — 1 dB gap from VP8 format floor)

### JPEG Encoder (rasmcore-jpeg)

| Aspect | Reference | Version | Result |
|--------|-----------|---------|--------|
| Baseline decode | libjpeg-turbo (djpeg) | 3.1.3 | All outputs decodable |
| Progressive decode | image crate (Rust) | 0.25 | All outputs decodable |
| Huffman tables | ITU-T T.81 Annex K | N/A | Standard tables K.3/K.5 |
| Quantization tables | mozjpeg Robidoux preset | N/A | Matched from mozjpeg source |
| Trellis lambda | mozjpeg formula | N/A | `2^14.75 / (2^16.5 + block_energy)` |
| DCT (shared crate) | HEVC spec ITU-T H.265 | N/A | Butterfly with E8/O8 constants |
| Three-way parity | mozjpeg + libjpeg-turbo | 3.1.3 | SSIM > 0.99 at matched QP |

### HEVC Decoder (rasmcore-hevc)

| Aspect | Reference | Version | Result |
|--------|-----------|---------|--------|
| CABAC engine | ITU-T H.265 Section 9 | N/A | Context model + bypass mode |
| DCT/DST transforms | ITU-T H.265 Table 8-6 | N/A | Butterfly constants verified |

### PNG Encoder/Decoder (via image crate)

| Aspect | Reference | Version | Result |
|--------|-----------|---------|--------|
| Pixel-exact roundtrip | image crate | 0.25 | Encode → decode = identical pixels |
| vs ImageMagick | ImageMagick | 7.1.2-18 | Pixel-exact for all filter/compression combos |

---

## Filter Validation (rasmcore-image)

See [crates/rasmcore-image/REFERENCE.md](../crates/rasmcore-image/REFERENCE.md)
for per-filter details including alignment notes.

### Pixel-Exact (MAE = 0.0000)

| Operation | Reference | Tool Version |
|-----------|-----------|-------------|
| premultiply | Python integer formula | Python 3.14 |
| add/remove alpha | Self roundtrip | N/A |
| convolve (identity) | Self | N/A |
| convolve (sharpen 3x3) | OpenCV `filter2D` | 4.13.0 |
| convolve (box blur 3x3) | OpenCV `blur` | 4.13.0 |
| median (r=1) | Pillow `MedianFilter` | 12.1.1 |
| sobel | OpenCV `Sobel` L2 | 4.13.0 |
| sepia | numpy matrix multiply | 2.4.3 |
| blend Multiply | numpy W3C CSS L1 | 2.4.3 |
| blend Screen | numpy W3C CSS L1 | 2.4.3 |
| bilateral filter | OpenCV `bilateralFilter` | 4.13.0 |
| brightness(0) | Self identity | N/A |

### Deterministic (MAE < 0.1, max_err = 1)

| Operation | Reference | Tool Version | Why Not Exact |
|-----------|-----------|-------------|---------------|
| CLAHE | OpenCV `createCLAHE` | 4.13.0 | f32 bilinear interpolation rounding |
| guided filter | OpenCV `ximgproc.guidedFilter` | 4.13.0 | f32 summed-area-table rounding |

---

## Transform Validation

| Operation | Reference | Tool Version | Result |
|-----------|-----------|-------------|--------|
| resize (Lanczos) | ImageMagick | 7.1.2-18 | MAE < 1.0 |
| crop | ImageMagick | 7.1.2-18 | Pixel-exact |
| rotate 90/180/270 | ImageMagick | 7.1.2-18 | Pixel-exact |
| flip H/V | ImageMagick | 7.1.2-18 | Pixel-exact |
| grayscale | BT.709 fixed-point | N/A | Integer formula verified |

---

## Color Conversion Validation

| Operation | Reference | Standard | Result |
|-----------|-----------|----------|--------|
| RGB to YCbCr (BT.601) | rasmcore-color roundtrip | ITU-R BT.601 | MAE ≤ 2 (integer rounding) |
| RGB to YCbCr (BT.709) | rasmcore-color roundtrip | ITU-R BT.709 | MAE ≤ 2 |
| RGB to YCbCr (BT.2020) | rasmcore-color roundtrip | ITU-R BT.2020 | MAE ≤ 2 |
| Grayscale (to_grayscale) | BT.601 `(77R+150G+29B+128)>>8` | ITU-R BT.601 | Integer-exact |

---

## Test Infrastructure

### Python Venv

Location: `tests/fixtures/.venv/`

```bash
python3 -m venv tests/fixtures/.venv
tests/fixtures/.venv/bin/pip install numpy==2.4.3 Pillow==12.1.1 opencv-python-headless==4.13.0.86
```

| Package | Pinned Version | Purpose |
|---------|---------------|---------|
| numpy | 2.4.3 | Formula reference (sepia, blend modes) |
| OpenCV | 4.13.0 | Spatial filter reference (convolve, sobel, bilateral, CLAHE) |
| Pillow | 12.1.1 | Median filter, format I/O reference |

### System Tools

| Tool | Version | Location | Purpose |
|------|---------|----------|---------|
| ImageMagick | 7.1.2-18 Q16-HDRI | `/opt/homebrew/bin/magick` | Fixture generation, transform parity |
| libwebp | 1.6.0 | `/opt/homebrew/bin/cwebp`, `dwebp` | VP8 bitstream validation |
| libjpeg-turbo | 3.1.3 | `/opt/homebrew/bin/cjpeg`, `djpeg` | JPEG bitstream validation |
| libvips | 8.18.1 | `/opt/homebrew/bin/vips` | Performance benchmarking |

### Test Files

| File | Scope | Tests |
|------|-------|-------|
| `crates/rasmcore-image/tests/filter_reference_parity.rs` | Convolution, color, blend | 13 tests, venv Python |
| `crates/rasmcore-image/tests/opencv_parity.rs` | Bilateral, CLAHE, guided | 21+ tests, venv Python |
| `crates/rasmcore-image/tests/reference_audit.rs` | All ops vs ImageMagick | 30+ tests, Docker ImageMagick |
| `crates/rasmcore-image/tests/parity.rs` | Decode/encode/transform | 20+ tests, fixtures from generate.sh |
| `crates/rasmcore-webp/tests/encode_decode.rs` | VP8 encode → decode | 11 tests, image-webp decoder |
| `crates/rasmcore-jpeg/tests/parity.rs` | JPEG three-way | 28+ tests, libjpeg-turbo + mozjpeg |

---

## Policy

1. **Every new operation must have a reference parity test** before merge
2. **Exact match (MAE=0.0) is the target** for all operations
3. **If exact match is impossible**, document why (f32 rounding, different algorithm) and set the tightest achievable threshold
4. **Reference tool versions are pinned** — version bumps require re-validation
5. **Tests never skip** — missing tools cause test failure with setup instructions
