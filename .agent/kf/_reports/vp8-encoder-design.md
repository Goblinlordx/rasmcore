# VP8 Encoder Design Document — rasmcore-webp

## Overview

This document specifies the architecture for `rasmcore-webp`, a pure Rust VP8
lossy encoder implementing RFC 6386 (VP8 Data Format and Decoding Guide). The
crate produces standard WebP files with lossy VP8 bitstreams, achieving quality
and file size parity with Google's libwebp reference implementation.

**Design principles:**
- Pure Rust, no C dependencies, compiles to wasm32-wasip2
- Each module is a reusable, well-abstracted component with a clean public API
- Deterministic output (same input → identical bytes)
- Integer arithmetic in hot paths (no floating point in encode loops)
- Correctness first, SIMD optimization later

---

## 1. VP8 Encoding Pipeline

```
Input Pixels (RGB/RGBA)
    │
    ▼
┌─────────────────┐
│  Color Convert   │  RGB → YUV420 (Y: full res, U/V: half res)
│  (color.rs)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Macroblock      │  Partition into 16×16 macroblocks
│  Iterator        │  Sub-blocks: 16 Y(4×4) + 4 U(4×4) + 4 V(4×4)
│  (block.rs)      │
└────────┬────────┘
         │
    ┌────┴────┐  (per macroblock)
    ▼         ▼
┌────────┐ ┌────────────┐
│Predict  │ │  Mode       │  Try all modes, pick lowest distortion
│(predict │ │  Selection  │  SAD/SATD metric for rate-distortion
│  .rs)   │ │             │
└────┬───┘ └─────────────┘
     │
     ▼
┌─────────────────┐
│  Residual        │  actual_pixels - predicted_pixels
│  Computation     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Forward DCT     │  4×4 integer DCT on residual blocks
│  (dct.rs)        │  4×4 WHT on Y DC coefficients
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Quantization    │  Scale coefficients by quality-derived step sizes
│  (quant.rs)      │  Six quantizer types: y_dc, y_ac, y2_dc, y2_ac, uv_dc, uv_ac
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Boolean Coder   │  Arithmetic encode quantized coefficients + modes
│  (boolcoder.rs)  │  Token tree with probability context
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Bitstream       │  Frame header + first partition + token partitions
│  Assembly        │  RIFF/WebP container wrapping
│  (bitstream.rs)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Loop Filter     │  Deblocking filter on reconstructed pixels
│  (filter.rs)     │  (applied during encode for reference frame quality)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rate Control    │  Quality (1-100) → quantizer index mapping
│  (ratecontrol.rs)│  Matches libwebp behavior for user expectations
└─────────────────┘
```

---

## 2. Module Structure — crates/rasmcore-webp/

```
src/
├── lib.rs              # Public API: encode(), EncodeConfig, EncodeError
├── config.rs           # EncodeConfig, quality presets
├── error.rs            # EncodeError enum
├── color.rs            # RGB→YUV420 conversion, YuvImage buffer
├── block.rs            # Macroblock/sub-block layout, iteration
├── predict.rs          # All 18 intra-prediction modes + mode selection
├── dct.rs              # 4×4 forward/inverse DCT, 4×4 WHT
├── quant.rs            # Quantization tables, quality→QP mapping
├── boolcoder.rs        # Boolean arithmetic encoder (reusable)
├── token.rs            # Coefficient token tree, probability context
├── bitstream.rs        # VP8 frame assembly, partition management
├── container.rs        # RIFF/WebP container wrapping
├── filter.rs           # Loop filter (simple + normal)
├── ratecontrol.rs      # Quality→quantizer mapping, filter strength
└── tables.rs           # Static lookup tables (dequant, norm, zigzag)
```

Each module exposes a clean public API designed for reuse:
- `boolcoder` can be used independently for any arithmetic coding task
- `dct` provides general-purpose 4×4 integer DCT/IDCT
- `color` provides RGB↔YUV conversion usable by other codecs
- `predict` provides prediction modes reusable for VP9 or similar

---

## 3. Module Specifications

### 3.1 Boolean Arithmetic Coder (`boolcoder.rs`)

**Reusable abstraction:** General-purpose boolean arithmetic encoder/decoder.

**Public API:**
```rust
pub struct BoolWriter {
    buf: Vec<u8>,
    range: u32,     // 0..=255
    value: u32,     // Accumulated bits
    run: u32,       // Pending 0xFF bytes (carry handling)
    nb_bits: i32,   // Bit accumulator position
}

impl BoolWriter {
    pub fn new() -> Self;
    pub fn with_capacity(capacity: usize) -> Self;
    pub fn put_bit(&mut self, prob: u8, bit: bool);
    pub fn put_bit_uniform(&mut self, bit: bool);
    pub fn put_literal(&mut self, n_bits: u8, value: u32);
    pub fn put_signed(&mut self, n_bits: u8, value: i32);
    pub fn finish(self) -> Vec<u8>;
    pub fn size_estimate(&self) -> usize;
}
```

**Algorithm (from RFC 6386 Section 7 + libwebp VP8PutBit):**
```
put_bit(prob, bit):
    split = (range * prob as u32) >> 8
    if bit:
        value += split + 1
        range -= split + 1
    else:
        range = split
    if range < 127:
        shift = NORM_TABLE[range]
        range = NEW_RANGE_TABLE[range]
        value <<= shift
        nb_bits += shift
        if nb_bits > 0: flush()
```

**C patterns avoided:**
- Raw pointer buffer management → `Vec<u8>` with automatic growth
- Global `kNorm`/`kNewRange` tables → `const` arrays in `tables.rs`
- Carry propagation via pointer arithmetic → `run` counter with deferred write

### 3.2 Integer DCT (`dct.rs`)

**Reusable abstraction:** 4×4 integer DCT/IDCT, 4×4 Walsh-Hadamard Transform.

**Public API:**
```rust
/// Forward 4×4 DCT on residual block (src - ref).
pub fn forward_dct(src: &[u8; 16], reference: &[u8; 16], out: &mut [i16; 16]);

/// Inverse 4×4 DCT, adding result to reference pixels.
pub fn inverse_dct(coeffs: &[i16; 16], reference: &[u8; 16], dst: &mut [u8; 16]);

/// Forward 4×4 Walsh-Hadamard Transform on DC coefficients.
pub fn forward_wht(dc_coeffs: &[i16; 16], out: &mut [i16; 16]);

/// Inverse WHT.
pub fn inverse_wht(coeffs: &[i16; 16], out: &mut [i16; 16]);
```

**Algorithm (from libwebp FTransform_C):**
```
Horizontal pass:
  d0..d3 = src[i] - ref[i] for each column
  a0 = d0 + d3, a1 = d1 + d2, a2 = d1 - d2, a3 = d0 - d3
  tmp[0] = (a0 + a1) * 8
  tmp[1] = (a2 * 2217 + a3 * 5352 + 1812) >> 9
  tmp[2] = (a0 - a1) * 8
  tmp[3] = (a3 * 2217 - a2 * 5352 + 937) >> 9

Vertical pass: similar butterfly structure on tmp columns
```

Constants 2217 and 5352 are fixed-point approximations of cos/sin for the
rotation step. All arithmetic is integer — no floating point.

**C patterns avoided:**
- BPS stride constant for pointer arithmetic → explicit 4×4 array indexing
- Function pointer dispatch for SIMD → direct functions, SIMD via `cfg` later

### 3.3 Quantization (`quant.rs`)

**Public API:**
```rust
pub struct QuantMatrix {
    pub q: [u16; 16],       // Quantizer step sizes
    pub iq: [u16; 16],      // Inverse quantizer (1/q fixed-point)
    pub bias: [u32; 16],    // Rounding bias
    pub zthresh: [u16; 16], // Zero threshold (skip if coeff < this)
    pub sharpen: [i16; 16], // Sharpening offsets
}

/// Build quantization matrix from QP index (0-127) and type.
pub fn build_matrix(qp: u8, qtype: QuantType) -> QuantMatrix;

/// Quantize a 4×4 block, returning index of last non-zero coefficient.
pub fn quantize_block(coeffs: &[i16; 16], matrix: &QuantMatrix, out: &mut [i16; 16]) -> i32;

/// Dequantize a 4×4 block.
pub fn dequantize_block(quantized: &[i16; 16], matrix: &QuantMatrix, out: &mut [i16; 16]);

/// Map user quality (1-100) to QP index (0-127), matching libwebp.
pub fn quality_to_qp(quality: u8) -> u8;
```

**Quality-to-QP mapping (from libwebp quant_enc.c):**
```
compression = quality_to_compression(quality / 100.0)
  where: if c < 0.75: linear_c = c * 2/3
         else:        linear_c = 2*c - 1
  compression = linear_c^(1/3)

qp = clamp(127 * (1 - compression), 0, 127)
```

**Dequantization tables:** `kDcTable[128]` and `kAcTable[128]` from RFC 6386
Section 20.3 — fixed lookup tables mapping QP index to step size.

**C patterns avoided:**
- `VP8Matrix` struct with raw arrays → typed `QuantMatrix` with named fields
- Zigzag scan via global array → `const ZIGZAG: [usize; 16]` in `tables.rs`
- `QUANTDIV` macro → inline function with explicit types

### 3.4 Prediction (`predict.rs`)

**Reusable abstraction:** Block prediction functions usable for VP8/VP9.

**Public API:**
```rust
pub enum Intra16Mode { DC, V, H, TM }
pub enum Intra4Mode { DC, TM, V, H, LD, RD, VR, VL, HD, HU }
pub enum ChromaMode { DC, V, H, TM }

/// Predict a 16×16 luma block from neighbors.
pub fn predict_16x16(mode: Intra16Mode, above: &[u8; 16], left: &[u8; 16],
                     above_left: u8, dst: &mut [u8; 256]);

/// Predict a 4×4 luma sub-block from neighbors.
pub fn predict_4x4(mode: Intra4Mode, above: &[u8; 4], left: &[u8; 4],
                   above_left: u8, above_right: &[u8; 4], dst: &mut [u8; 16]);

/// Predict an 8×8 chroma block from neighbors.
pub fn predict_8x8(mode: ChromaMode, above: &[u8; 8], left: &[u8; 8],
                   above_left: u8, dst: &mut [u8; 64]);

/// Select best prediction mode using SAD distortion metric.
pub fn select_best_16x16(actual: &[u8; 256], above: &[u8; 16], left: &[u8; 16],
                         above_left: u8) -> Intra16Mode;

/// Sum of absolute differences between two blocks.
pub fn sad(a: &[u8], b: &[u8]) -> u32;
```

**Prediction formulas (from RFC 6386 Section 12):**
- DC: fill with `(sum(above) + sum(left) + count) / (2*count)`
- V: copy `above[x]` to every row
- H: copy `left[y]` to every column
- TM: `above[x] + left[y] - above_left`, clamped to [0, 255]
- Diagonal modes (4×4): `(a + 2*b + c + 2) >> 2` weighted averages

**C patterns avoided:**
- Function pointer tables for mode dispatch → match on enum
- `BPS` stride for pixel access → explicit slice parameters with known sizes
- In-place pixel modification → separate `dst` output buffer

### 3.5 Bitstream Assembly (`bitstream.rs` + `container.rs`)

**Frame structure (from RFC 6386):**
```
[Frame Tag: 3 bytes]
  Bit 0: keyframe (0)
  Bits 1-3: version (0)
  Bit 4: show_frame (1)
  Bits 5-23: first_partition_size (19 bits, little-endian)

[Keyframe Header: 7 bytes]
  Bytes 0-2: start code 0x9D 0x01 0x2A
  Bytes 3-4: width | (hscale << 14)  (little-endian)
  Bytes 5-6: height | (vscale << 14) (little-endian)

[First Partition: bool-coded]
  color_space (1 bit), clamping_type (1 bit)
  segmentation header, filter header, partition count
  per-macroblock: prediction modes

[Token Partitions: 1-8, each bool-coded]
  per-macroblock: DCT coefficients via token tree
```

**WebP RIFF container (from WebP Container Spec):**
```
[RIFF] [file_size: u32 LE] [WEBP]
[VP8 ] [chunk_size: u32 LE] [vp8_frame_data]
```

Padding to even byte boundary if chunk data is odd-length.

### 3.6 Loop Filter (`filter.rs`)

**Public API:**
```rust
pub enum FilterType { Simple, Normal }

/// Apply loop filter to reconstructed frame.
pub fn apply_loop_filter(pixels: &mut YuvImage, filter_level: u8,
                         sharpness: u8, filter_type: FilterType,
                         mb_info: &[MacroblockInfo]);
```

**Simple filter:** Adjusts 2 pixels on each side of macroblock edges.
**Normal filter:** Adjusts 4 pixels on each side, with high-edge-variance detection.

Filter strength per macroblock: `base_level + mode_delta + ref_delta`, clamped to [0, 63].

### 3.7 Rate Control (`ratecontrol.rs`)

**Public API:**
```rust
/// Complete quality-to-encoder-params mapping.
pub fn quality_to_params(quality: u8) -> EncodeParams;

pub struct EncodeParams {
    pub qp_y: u8,           // Luma QP index
    pub qp_uv: u8,          // Chroma QP index
    pub filter_level: u8,   // Loop filter strength
    pub filter_sharpness: u8,
    pub filter_type: FilterType,
}
```

---

## 4. C-Specific Patterns → Rust Alternatives

| C Pattern (libwebp) | Rust Alternative |
|---|---|
| Raw pointer arithmetic for pixel access (`ptr + y*BPS + x`) | Slice indexing with known sizes (`block[y*4 + x]`) or fixed-size arrays (`[u8; 16]`) |
| Global mutable state (`VP8Encoder` struct with everything) | Per-module state passed explicitly; `Encoder` struct composes modules |
| Manual buffer reallocation (`realloc`) | `Vec<u8>` with automatic growth |
| Function pointer dispatch tables (SIMD) | Direct functions + `#[cfg]` for SIMD variants later |
| `kNorm`/`kNewRange` global lookup tables | `const` arrays in `tables.rs` module |
| Macro-heavy code (`QUANTDIV`, `STORE`, etc.) | Inline functions with explicit types |
| goto-based error handling | `Result<T, E>` with `?` operator |
| Carry propagation via pointer backtrack | `run` counter with deferred byte emission |
| Platform-specific SIMD intrinsics (SSE, NEON) | Phase 1: scalar only. Future: `std::simd` portable SIMD or `wide` crate |
| `#define BPS` stride constant | Typed stride parameters or fixed-size array types |

---

## 5. Public API — crates/rasmcore-webp/src/lib.rs

```rust
/// Encode raw pixels to lossy WebP (VP8).
///
/// Accepts RGB8 or RGBA8 pixel data. Returns a complete WebP file
/// (RIFF container with VP8 chunk).
pub fn encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    format: PixelFormat,
    config: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError>;

/// Pixel format of input data.
pub enum PixelFormat {
    Rgb8,
    Rgba8,
}

/// Encoder configuration.
pub struct EncodeConfig {
    /// Quality level 1-100. Maps to VP8 quantizer index matching
    /// libwebp behavior so users get predictable results.
    pub quality: u8,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        Self { quality: 75 }
    }
}

/// Encoding error.
pub enum EncodeError {
    InvalidDimensions { width: u32, height: u32 },
    InvalidPixelData { expected: usize, actual: usize },
    EncodeFailed(String),
}
```

---

## 6. Dependency Analysis

**External crates needed:** None for core encoding. All algorithms are
implemented from the specification using integer arithmetic.

**Optional dev-dependencies for testing:**
- `image` — decode test images and reference WebP files
- `butteraugli` — perceptual quality comparison
- `dssim-core` — structural similarity comparison

**Internal dependencies:**
- `tables.rs` — imported by `boolcoder`, `dct`, `quant`, `token`
- `color.rs` — imported by top-level `encode()`
- `block.rs` — imported by top-level encoder loop
- `boolcoder.rs` — imported by `bitstream`
- `dct.rs` — imported by encoder loop
- `quant.rs` — imported by encoder loop and `ratecontrol`
- `predict.rs` — imported by encoder loop
- `token.rs` — imported by `bitstream`
- `filter.rs` — imported by encoder reconstruction loop
- `bitstream.rs` + `container.rs` — imported by top-level `encode()`
- `ratecontrol.rs` — imported by top-level `encode()`

---

## 7. Complexity Estimates

| Module | Relative Complexity | Key Challenge |
|---|---|---|
| `boolcoder.rs` | Low | Carry propagation logic |
| `dct.rs` | Low | Fixed-point constant accuracy |
| `quant.rs` | Medium | Quality-to-QP curve matching libwebp |
| `predict.rs` | Medium | 10 diagonal modes (4×4), correct neighbor access |
| `token.rs` | Medium | Probability context state machine |
| `bitstream.rs` | High | Connecting all pieces, partition management |
| `container.rs` | Low | Simple RIFF wrapping |
| `filter.rs` | Medium | Edge detection + multi-pixel filtering |
| `ratecontrol.rs` | Medium | Matching libwebp quality curve |
| `color.rs` | Low | Standard RGB↔YUV420 conversion |
| `block.rs` | Low | Layout and iteration helpers |
| `tables.rs` | Low | Static data transcription |

**Total estimated LOC:** ~3,000-4,000 lines of Rust (excluding tests).

---

## 8. SIMD Strategy (Future)

Phase 1 (these tracks): scalar-only implementation for correctness.

Phase 2 (future track): SIMD optimization using `std::simd` portable SIMD
or the `wide` crate (safe SIMD abstraction). Target functions:
- `forward_dct` / `inverse_dct` — 4×4 butterfly operations
- `predict_*` — row/column fill operations
- `quantize_block` — parallel multiply and threshold
- `sad` — sum of absolute differences
- `loop_filter` — pixel clamping operations

WASM target: SIMD128 instructions (already used by fast_image_resize and
libblur in this project). Auto-vectorization handles many cases in release
builds without explicit SIMD.
