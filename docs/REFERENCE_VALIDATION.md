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

### Inpainting — Telea + Navier-Stokes

| Method | Test Case | Reference | MAE | Max Error | Status |
|--------|-----------|-----------|-----|-----------|--------|
| Telea (FMM) | uniform 16x16, 4x4 hole | OpenCV `cv2.inpaint(..., INPAINT_TELEA)` 4.13.0 | 0.0000 | 0 | Exact |
| NS (FMM) | uniform 16x16, 4x4 hole | OpenCV `cv2.inpaint(..., INPAINT_NS)` 4.13.0 | 0.0000 | 0 | Exact |
| NS (FMM) | gradient 32x32, 6x6 hole | OpenCV `cv2.inpaint(..., INPAINT_NS)` 4.13.0 | 0.0000 | 0 | Exact |
| Telea (FMM) | gradient 32x32, 6x6 hole | OpenCV `cv2.inpaint(..., INPAINT_TELEA)` 4.13.0 | 0.0645 | 6 | See note |

**Telea gradient note:** max_err=6 (2.4% of 255) on non-uniform content. Root-caused
to f32 accumulation order in the gradient correction term `(Jx+Jy)/|J|`. A standalone
C replica of OpenCV's published source (exact same algorithm, same f32 types, same
accumulation order) produces the same result as our Rust implementation — not the
same as the OpenCV 4.13.0 binary. The binary likely differs due to compiler
optimizations (FMA instructions, extended-precision registers) that change f32
intermediate rounding. This affects even single-pixel inpainting on gradient images
(ours=107, OpenCV binary=108, our C replica=107). The NS method (pure weighted
average, no gradient correction) is unaffected and achieves exact match.

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

## Reference Selection Rationale

Not all tools are suitable references for all operations. This section
documents why specific tools were chosen — and why others were rejected —
for each category of validation.

### Why OpenCV over Pillow for precision-sensitive operations

**Pillow (PIL) has known precision deficiencies** that make it unsuitable
as a reference for operations requiring mathematical correctness:

| Deficiency | Details | Impact |
|------------|---------|--------|
| **16→8 bit-depth uses truncation** | Pillow's `Image.convert('RGB')` on 16-bit images uses `v >> 8` (right shift / truncation) instead of the mathematically correct `round(v * 255 / 65535)`. | 26% of values differ by 1 vs correct rounding. Example: u16=1023 → Pillow gives 3 (`1023>>8`), correct is 4 (`round(1023*255/65535)`). |
| **16-bit RGB not natively supported** | Pillow reads 16-bit RGB PNG as mode `'RGB'` (8-bit) or `'I;16'` (single-channel). There is no native 16-bit RGB mode. `Image.fromarray()` rejects `uint16` RGB arrays. | Cannot use Pillow for 16-bit RGB pixel comparison without lossy conversion. |
| **Q8 internal processing** | Pillow processes images at 8-bit precision internally, even when the source is 16-bit. Operations like histogram equalization, gamma, and color adjustment lose the low 8 bits. | Comparing 16-bit processing output against Pillow always shows artificial MAE from Pillow's precision loss, not from our implementation. |

**OpenCV** operates at the requested bit depth (8, 16, or 32-bit float)
and uses `round()` for bit-depth conversion (matching the IEEE 754
standard). Our `(v + 128) / 257` formula is algebraically identical to
OpenCV's `saturate_cast<uchar>(v * 255.0 / 65535.0 + 0.5)`.

**numpy** uses f64 precision for formula computation, making it the gold
standard for validating mathematical formulas (gamma, sepia matrix, blend
modes). Our f64 `powf()` matches numpy's f64 `**` operator exactly.

**ImageMagick Q16-HDRI** processes at 64-bit float precision internally.
When comparing at native Q16 (via `magick ... rgb:-` raw output), our f64
gamma formula produces identical results. Previous tests that routed IM
output through Pillow's 8-bit `.convert('RGB')` showed false MAE=0.10
from Pillow's lossy `>>8` conversion — not from any algorithmic difference.

### Where Pillow IS a valid reference

Pillow remains valid for:
- **Median filter** (sorting-based, integer-only, pixel-exact at 8-bit)
- **Format I/O** (PNG/TIFF decode at 8-bit precision)
- **Image creation** (test fixture generation)

Pillow is NOT valid for:
- **Bit-depth conversion** (uses truncation instead of rounding)
- **16-bit processing** (downcasts to 8-bit internally)
- **Precision-sensitive formulas** (use numpy instead)

### Reference hierarchy

When multiple tools could validate an operation, prefer in this order:

1. **numpy** — for pure formula validation (MAE=0.0 achievable)
2. **OpenCV** — for spatial filters, bit-depth conversion, image processing
3. **ImageMagick Q16-HDRI** — for full-pipeline validation (compare at native Q16 via `rgb:-`)
4. **Pillow** — only for 8-bit integer operations (median, format I/O)
5. **Self-validation** — roundtrip tests, identity kernels, known input/output pairs

### Lesson learned

When comparing against a reference, **always compare at native precision**.
Routing high-precision output through a lower-precision tool (e.g., reading
16-bit IM output via Pillow's 8-bit `convert('RGB')`) creates false
divergences that are artifacts of the comparison pathway, not of the
implementation under test. We discovered this when:

- Gamma at Q16 showed MAE=0.10 via Pillow (false) but MAE=0.0000 via raw `rgb:-` (true)
- 16→8 conversion showed MAE=0.26 vs Pillow (Pillow's formula is wrong) but MAE=0.0 vs OpenCV (correct)

---

## Test Infrastructure

### Python Venv

Location: `tests/fixtures/.venv/`

```bash
python3 -m venv tests/fixtures/.venv
tests/fixtures/.venv/bin/pip install numpy==2.4.3 Pillow==12.1.1 opencv-python-headless==4.13.0.86
```

| Package | Pinned Version | Purpose | Precision |
|---------|---------------|---------|-----------|
| numpy | 2.4.3 | Formula reference (sepia, blend modes, gamma) | f64 — gold standard |
| OpenCV | 4.13.0 | Spatial filters, bit-depth conversion, bilateral, CLAHE | Native bit depth |
| Pillow | 12.1.1 | 8-bit median filter, format I/O only (see deficiencies above) | 8-bit only |

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
