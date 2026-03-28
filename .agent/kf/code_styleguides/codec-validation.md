# Codec Validation Standard

Every native codec implementation MUST include three-way reference validation
before a track can be marked complete. This applies to ALL codec crates
(rasmcore-qoi, rasmcore-pnm, rasmcore-bmp, rasmcore-tga, rasmcore-fits,
rasmcore-webp, rasmcore-jpeg, and any future codecs).

## Three-Way Validation

For each supported encode format/variant, run this pipeline:

```
A = our_encode(original) → our_decode
B = our_encode(original) → ref_decode
C = ref_encode(original) → ref_decode
```

Where `ref` is a known-good reference implementation (typically the `image` crate,
ImageMagick, or the canonical format library).

### Assertions

**Lossless codecs** (QOI, PNM, BMP, TGA, FITS, PNG, GIF, etc.):
- `B == original` — bit-exact. Our encoded output, decoded by reference, recovers the original perfectly.
- `A == B` — bit-exact. Our decoder produces the same pixels as the reference decoder.
- `C == original` — bit-exact. Sanity check that the reference roundtrips correctly.
- Any deviation in ANY of these is a bug that must be fixed before the track is complete.

**Lossy codecs** (JPEG, WebP, AVIF, etc.):
- `C_quality = psnr(C, original)` — establishes the reference quality baseline.
- `B_quality = psnr(B, original)` — our encoder's quality.
- `B_quality >= C_quality * 0.9` — our output must be at least 90% as good as the reference at the same quality setting.
- `A == B` — our decoder and the reference decoder produce identical pixels from our encoded output.
- For JPEG specifically: PSNR > 30dB at quality 85, PSNR > 25dB at quality 50.

### What To Test

Each codec must test ALL supported encode variants:

| Variant type | Examples |
|-------------|----------|
| Bit depths | 1-bit, 4-bit, 8-bit, 16-bit, 32-bit float |
| Color modes | Grayscale, RGB, RGBA, indexed/palette |
| Compression | Uncompressed, RLE, Huffman, arithmetic |
| Subsampling | 4:4:4, 4:2:2, 4:2:0 (JPEG) |
| Container | JFIF, RIFF/WebP, FITS HDU |

### Test Images

Use at minimum these patterns for each variant:
- **Solid color** (1×1 and 16×16) — baseline sanity
- **Gradient** (32×32) — exercises all encoder paths (prediction, DCT, entropy)
- **Checkerboard** (16×16 with 4×4 squares) — stresses RLE and block boundaries
- **Odd dimensions** (3×7, 17×1) — catches row padding and MCU boundary bugs

### Where Tests Live

Parity tests go in each codec crate's `tests/` directory (integration tests),
not in `src/` (unit tests). They are separate from the internal roundtrip tests
because they require `image` crate as a dev-dependency.

```
crates/rasmcore-{codec}/tests/parity.rs
```

### When This Applies

- Every new codec track MUST include parity tests as acceptance criteria.
- Every new encode variant added to an existing codec MUST add corresponding parity tests.
- Architects MUST include "Three-way reference validation for all variants" in track acceptance criteria.
- Developers MUST NOT mark a codec track complete without passing parity tests.

### Reference Decoders

| Codec | Primary Reference | Fallback |
|-------|------------------|----------|
| QOI | `image` crate (ImageFormat::Qoi) | — |
| PNM | `image` crate (ImageFormat::Pnm) | ImageMagick `convert` |
| BMP | `image` crate (ImageFormat::Bmp) | — |
| TGA | `image` crate (ImageFormat::Tga) | — |
| JPEG | `image` crate (ImageFormat::Jpeg) | ImageMagick `convert` |
| WebP | `image` crate (ImageFormat::WebP) | `dwebp` CLI |
| FITS | Self-consistency only (no `image` support) | `cfitsio` if available |
| PNG | `image` crate (ImageFormat::Png) | — |

### Exception: Codecs Without Reference Decoders

For codecs where no reference implementation exists in the `image` crate (e.g., FITS),
use self-consistency testing (A == roundtrip) plus manual validation against CLI tools
when available. Document the limitation in the test file.
