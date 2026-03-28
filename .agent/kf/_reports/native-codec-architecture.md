# Native Codec Architecture — Design Document

**Date:** 2026-03-28
**Track:** codec-native-architecture_20260328070000Z

---

## 1. Reference Implementation: rasmcore-webp

The WebP encoder (`crates/rasmcore-webp/`) is the proof-of-concept for native codec ownership.

### Architecture

- **3,488 LOC** across 14 modules, **zero external dependencies**
- Separate crate from rasmcore-image (clean dependency boundary)
- Modular design: each algorithm is an independent, testable module

### Module Map

| Module | LOC | Purpose |
|--------|-----|---------|
| boolcoder.rs | 451 | Boolean arithmetic encoder/decoder (VP8 entropy) |
| predict.rs | 823 | VP8 intra-prediction (4x4, 8x8, 16x16 modes) |
| quant.rs | 508 | Quantization tables and quality mapping |
| bitstream.rs | 525 | VP8 frame assembly |
| dct.rs | 444 | 4x4 integer DCT/IDCT + Walsh-Hadamard |
| token.rs | 243 | Coefficient token tree encoding |
| color.rs | 153 | RGB to YUV420 conversion |
| container.rs | 65 | RIFF/WebP wrapper |
| Other (6) | 279 | Config, error, tables, stubs |

### Test Patterns (100+ tests)

1. **Roundtrip**: forward(inverse(x)) == x (DCT, bool coder, quant)
2. **Reference values**: Snapshot expected output for regression detection
3. **Exhaustive**: 100-1000 iterations with pseudo-random input
4. **Property**: Monotonicity, determinism, range constraints
5. **Integration**: Full pipeline (DCT -> quant -> dequant -> IDCT)
6. **Performance**: 1M boolean encodings in <1s

### Key Takeaways

- Zero dependencies works. Pure Rust compiles to any target including wasm32-wasip2.
- Each algorithm module is independently testable — no need to wire the full codec for unit testing.
- ~3,500 LOC for a complete lossy image encoder is manageable.
- The module boundary pattern (boolcoder, dct, predict, quant) maps to spec sections.

---

## 2. Common Codec Trait Design

### Encoder Trait

```rust
/// Configuration for an encoder. Each codec defines its own config type.
pub trait EncodeConfig: Default + Clone {}

/// Core encoder trait — all rasmcore native codecs implement this.
pub trait ImageEncoder {
    type Config: EncodeConfig;
    type Error: std::error::Error;

    /// Encode raw pixels to the codec's format.
    fn encode(
        pixels: &[u8],
        width: u32,
        height: u32,
        format: PixelFormat,
        config: &Self::Config,
    ) -> Result<Vec<u8>, Self::Error>;

    /// Supported input pixel formats.
    fn supported_formats() -> &'static [PixelFormat];
}
```

### Decoder Trait

```rust
/// Core decoder trait.
pub trait ImageDecoder {
    type Error: std::error::Error;

    /// Detect if this decoder handles the given data.
    fn detect(header: &[u8]) -> bool;

    /// Decode to pixels + metadata.
    fn decode(data: &[u8]) -> Result<DecodedOutput, Self::Error>;

    /// Decode header only (dimensions, format) without pixel allocation.
    fn probe(data: &[u8]) -> Result<ImageInfo, Self::Error>;
}

pub struct DecodedOutput {
    pub pixels: Vec<u8>,
    pub info: ImageInfo,
    pub metadata: MetadataSet,
}
```

### Integration with Existing Domain API

The domain encoder interface (`encode(pixels, info, config) -> Result<Vec<u8>>`) stays stable.
Native codecs implement `ImageEncoder`, and the domain `encoder/xyz.rs` module wraps them:

```rust
// domain/encoder/qoi.rs (example)
pub fn encode(pixels: &[u8], info: &ImageInfo, config: &QoiEncodeConfig) -> Result<Vec<u8>, ImageError> {
    rasmcore_qoi::QoiEncoder::encode(pixels, info.width, info.height, map_format(info.format), config)
        .map_err(|e| ImageError::ProcessingFailed(e.to_string()))
}
```

---

## 3. Codec Feasibility Classification

### Tier 1: TRIVIAL (own now, days each)

| Format | LOC | Algorithms | Shared Infra | Notes |
|--------|-----|-----------|--------------|-------|
| PNM/PPM | 200-400 | Header + raw pixels | None | Simplest possible format |
| QOI | 300-650 | Hash + diff + RLE | None | 1-page spec, public domain |
| BMP | 500-1,200 | Headers + RLE | None | Multiple header versions |
| TGA | 400-800 | Header + RLE | None | Simple, well-defined |
| ICO | 300-600 | Container only | BMP + PNG | Thin wrapper |

**Total: ~1,700-3,650 LOC. Estimated effort: 1-2 weeks for all 5.**

### Tier 2: MODERATE (own soon, weeks each)

| Format | LOC | Algorithms | Shared Infra | Notes |
|--------|-----|-----------|--------------|-------|
| GIF | 1,500-3,000 | LZW + palette + animation | LZW module | Animation compositing is the hard part |
| PNG | 4,000-6,000 | Deflate + filtering + interlacing | Deflate, CRC32, Huffman | Deflate is ~3K LOC alone |
| DDS | 3,000-8,000 | BCn block compression | None | BC1-5 moderate, BC6H/BC7 complex |

**Total: ~8,500-17,000 LOC. Estimated effort: 2-3 months.**

### Tier 3: COMPLEX (own later, months each)

| Format | LOC | Algorithms | Shared Infra | Notes |
|--------|-----|-----------|--------------|-------|
| JPEG | 8,000-15,000 | DCT + Huffman + YCbCr + quant | DCT, Huffman, color conv | zenjpeg is a head start |
| TIFF | 10,000-25,000 | IFD + multi-codec | LZW, Deflate, PackBits, JPEG | "Thousands of Incompatible File Formats" |

**Total: ~18,000-40,000 LOC. Estimated effort: 6-12 months.**

### Tier 4: MASSIVE (own container, wrap codec)

| Format | Container LOC | Codec LOC | Strategy |
|--------|-------------|-----------|----------|
| AVIF | 2,000-4,000 (ISOBMFF) | 40,000-80,000 (AV1) | Own ISOBMFF container, wrap rav1e/dav1d |
| HEIF/HEIC | Shared ISOBMFF | 40,000-70,000 (HEVC) | Own ISOBMFF container, wrap x265/libde265 (sidecar) |
| JPEG XL | 2,000-4,000 (JXL boxes) | 60,000-100,000 (codec) | Own container, wrap libjxl (or defer) |

**Strategy: Own the container format (ISOBMFF, JXL box structure) and treat the underlying codec as a pluggable backend. One ISOBMFF implementation serves both AVIF and HEIF.**

---

## 4. Shared Infrastructure Modules

These modules serve multiple codecs and should be implemented as shared crates.

| Module | LOC | Used By | Priority |
|--------|-----|---------|----------|
| **rasmcore-deflate** | 1,500-3,500 | PNG, TIFF, zlib streams | High (unlocks PNG + TIFF) |
| **rasmcore-huffman** | 200-500 | JPEG, Deflate | High (feeds into deflate + JPEG) |
| **rasmcore-lzw** | 300-800 | GIF, TIFF | Medium |
| **rasmcore-color** | 50-150 | JPEG, WebP, AVIF, HEIF | High (already exists in rasmcore-webp) |
| **rasmcore-dct** | 100-500 (8x8) | JPEG, WebP | Medium (WebP already has 4x4) |
| **rasmcore-crc32** | 50-100 | PNG, TIFF | Low (trivial) |
| **rasmcore-isobmff** | 2,000-4,000 | AVIF, HEIF | Deferred (Tier 4) |
| **rasmcore-bitio** | 200-400 | All codecs with bitstream parsing | High |
| **rasmcore-rle** | 50-100 | BMP, TGA, TIFF PackBits | Low (trivial) |

### Dependency Graph

```
rasmcore-bitio (bit reader/writer)
    |
    +-- rasmcore-huffman (Huffman coding)
    |       |
    |       +-- rasmcore-deflate (deflate/inflate = LZ77 + Huffman)
    |               |
    |               +-- PNG native codec
    |               +-- TIFF native codec (deflate mode)
    |
    +-- rasmcore-lzw (LZW compression)
    |       |
    |       +-- GIF native codec
    |       +-- TIFF native codec (LZW mode)
    |
    +-- rasmcore-dct (DCT transforms)
            |
            +-- JPEG native codec (8x8 fixed)
            +-- WebP (already has 4x4)

rasmcore-color (color space conversion)
    |
    +-- JPEG, WebP, AVIF, HEIF

rasmcore-isobmff (ISO Base Media File Format)
    |
    +-- AVIF container
    +-- HEIF container
```

---

## 5. Migration Strategy

### Phase Model: Feature Flags

```toml
# Cargo.toml
[features]
default = ["image-codecs"]  # Use image crate (backward compat)
image-codecs = ["image"]     # Third-party backends
native-qoi = []              # Pure rasmcore QOI
native-bmp = []              # Pure rasmcore BMP
native-png = ["rasmcore-deflate"]  # Pure rasmcore PNG
native-jpeg = ["rasmcore-dct"]     # Pure rasmcore JPEG
native-all = ["native-qoi", "native-bmp", "native-png", ...]
```

### Migration Path Per Codec

1. **Create rasmcore-xyz crate** (separate from rasmcore-image)
2. **Implement encoder + decoder** with full test suite
3. **Wire into domain/encoder/xyz.rs** behind feature flag
4. **Parity test**: native output vs image crate output (must match or improve)
5. **Benchmark**: native vs image crate (must not regress)
6. **Flip default**: native becomes default, image crate becomes fallback
7. **Remove image crate path** once native is stable

### Example: QOI Migration

```rust
// domain/encoder/qoi.rs
#[cfg(feature = "native-qoi")]
pub fn encode(pixels: &[u8], info: &ImageInfo, config: &QoiEncodeConfig) -> Result<Vec<u8>, ImageError> {
    rasmcore_qoi::encode(pixels, info.width, info.height, ...)
}

#[cfg(not(feature = "native-qoi"))]
pub fn encode(pixels: &[u8], info: &ImageInfo, config: &QoiEncodeConfig) -> Result<Vec<u8>, ImageError> {
    let img = pixels_to_dynamic_image(pixels, info)?;
    encode_via_image_format(&img, image::ImageFormat::Qoi)
}
```

---

## 6. Testing Philosophy

### Five-Layer Testing Strategy

#### Layer 1: Unit Tests (per algorithm module)
- Roundtrip: forward(inverse(x)) == x
- Reference values: snapshot expected output
- Property: monotonicity, determinism, range constraints
- Exhaustive: pseudo-random inputs over many iterations
- **Target: 100% coverage of algorithm modules**

#### Layer 2: Spec Compliance Tests
- Use official test suites where available:
  - PNG: PngSuite (http://www.schaik.com/pngsuite/)
  - JPEG: ITU-T T.83 conformance test data
  - TIFF: libtiff test images
  - GIF: GIF conformance samples
- Parse and decode every test vector; verify dimensions, pixel values, metadata
- **Target: pass all spec test vectors**

#### Layer 3: Parity Tests (vs reference implementations)
- Encode with rasmcore, decode with ImageMagick (and vice versa)
- Compare: pixel PSNR > threshold, file size within 10%
- Test across quality range (q10, q25, q50, q75, q90, q100)
- **Target: interoperability with ImageMagick and libvips**

#### Layer 4: Fuzzing
- cargo-fuzz on all decoder inputs
- AFL corpus from real-world images
- Ensure: no panics, no UB, graceful error on malformed input
- **Target: 0 crashes on fuzzer-generated inputs after 24h run**

#### Layer 5: Performance Benchmarks
- Criterion.rs benchmarks per codec
- Compare: native vs image crate vs ImageMagick
- Metrics: encode time, decode time, file size, peak memory
- **Target: native within 2x of image crate; within 5x of C implementations**

---

## 7. Container vs Codec Ownership

### Decision Matrix

| Format | Container | Codec Algorithm | Strategy |
|--------|-----------|----------------|----------|
| JPEG | JFIF/EXIF markers | DCT+Huffman | **Own both** (zenjpeg exists) |
| PNG | Chunk structure | Deflate+filtering | **Own both** (via rasmcore-deflate) |
| GIF | Block structure | LZW | **Own both** |
| TIFF | IFD structure | Multi-codec | **Own container + simple codecs** (LZW, PackBits, Deflate); wrap JPEG-in-TIFF |
| WebP | RIFF/VP8 | VP8 intra-frame | **Own both** (rasmcore-webp in progress) |
| AVIF | ISOBMFF | AV1 intra-frame | **Own container**, wrap rav1e/dav1d |
| HEIF | ISOBMFF | HEVC intra-frame | **Own container**, wrap x265/libde265 (sidecar) |
| JPEG XL | JXL boxes | VarDCT+Modular | **Own container**, wrap libjxl (or defer entirely) |

### Rationale

Writing a full AV1/HEVC/JXL codec from scratch is impractical (50K-100K LOC each, years of optimization). But owning the container gives us:
- Full control over metadata handling
- Custom ISOBMFF parsing for streaming/partial reads
- One ISOBMFF implementation serving both AVIF and HEIF
- The codec algorithm becomes a pluggable backend trait

---

## 8. Phased Implementation Roadmap

### Phase 1: Shared Infrastructure (2-3 weeks)

1. `rasmcore-bitio` — bit reader/writer (200-400 LOC)
2. `rasmcore-crc32` — CRC32 for PNG (50-100 LOC)
3. `rasmcore-color` — extract from rasmcore-webp, generalize (150 LOC)
4. `rasmcore-rle` — PackBits + BMP RLE + TGA RLE (100 LOC)

### Phase 2: Trivial Native Codecs (2-3 weeks)

5. `rasmcore-pnm` — PNM/PPM/PGM/PBM (200-400 LOC)
6. `rasmcore-qoi` — QOI (300-650 LOC)
7. `rasmcore-bmp` — BMP (500-1,200 LOC)
8. `rasmcore-tga` — TGA (400-800 LOC)
9. `rasmcore-ico` — ICO container (300-600 LOC, wraps BMP+PNG)

### Phase 3: Moderate Native Codecs (2-3 months)

10. `rasmcore-lzw` — shared LZW module (300-800 LOC)
11. `rasmcore-huffman` — shared Huffman coding (200-500 LOC)
12. `rasmcore-deflate` — deflate/inflate (1,500-3,500 LOC)
13. `rasmcore-gif` — GIF (1,500-3,000 LOC, uses rasmcore-lzw)
14. `rasmcore-png` — PNG (4,000-6,000 LOC, uses rasmcore-deflate)

### Phase 4: Complex Native Codecs (6-12 months)

15. `rasmcore-dct` — shared DCT module, generalize from rasmcore-webp (500 LOC)
16. `rasmcore-jpeg` — JPEG (8,000-15,000 LOC, uses DCT+Huffman+color)
17. TIFF native container + LZW/Deflate/PackBits modes (10,000-25,000 LOC)

### Phase 5: Container Ownership (deferred)

18. `rasmcore-isobmff` — ISOBMFF parser/writer for AVIF+HEIF containers
19. JXL box structure parser for JPEG XL container

### Not Planned: Full Codec Ownership

- AV1 encoding/decoding algorithm (wrap rav1e/dav1d)
- HEVC encoding/decoding algorithm (wrap x265/libde265, sidecar only)
- JPEG XL full codec (wrap libjxl or defer)

---

## 9. Effort Summary

| Phase | LOC Range | Timeline | Codecs |
|-------|----------|----------|--------|
| Shared infra | 500-1,100 | 2-3 weeks | Foundation for all |
| Trivial | 1,700-3,650 | 2-3 weeks | PNM, QOI, BMP, TGA, ICO |
| Moderate | 7,500-13,800 | 2-3 months | GIF, PNG (+ LZW, Huffman, Deflate) |
| Complex | 18,000-40,000 | 6-12 months | JPEG, TIFF |
| Containers | 4,000-8,000 | 1-2 months | ISOBMFF (AVIF/HEIF), JXL boxes |
| **Total own** | **~31,700-66,550** | **~12-18 months** | 10 codecs + 3 containers |
