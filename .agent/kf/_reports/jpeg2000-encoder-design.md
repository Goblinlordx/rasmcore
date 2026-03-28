# JPEG 2000 Support Design Document — rasmcore-jp2

## Overview

This document specifies the architecture for JPEG 2000 support in rasmcore,
covering both decode and encode. Based on ISO/IEC 15444-1 (ITU-T T.800).

**Key finding:** Multiple pure Rust JPEG 2000 implementations now exist.
Rather than implementing from scratch, we should evaluate and potentially
adopt one as a dependency or fork.

---

## 1. Pure Rust Ecosystem Assessment

### Available Crates (March 2026)

| Crate | Version | License | Pure Rust | Encode | Decode | Lossy | Lossless | WASM | Notes |
|-------|---------|---------|-----------|--------|--------|-------|----------|------|-------|
| **justjp2** | 0.1.1 | MIT/Apache-2.0 | Yes | Yes | Yes | Yes (9/7) | Yes (5/3) | Likely | 5K LOC, multi-tile, ROI, rayon parallel |
| **dicom-toolkit-jpeg2000** | 0.5.0 | MIT/Apache-2.0 | Yes | No? | Yes | Yes | Yes | Partial (no_std) | SIMD, 20K+ test images, HTJ2K via openjph |
| **openjp2** | 0.6.1 | ? | Yes (port) | Yes | Yes | Yes | Yes | ? | Rust port of OpenJPEG |
| **jpeg2k** | 0.10.1 | MIT/Apache-2.0 | No (wraps openjpeg-sys) | Yes | Yes | Yes | Yes | No (C dep) | Mature but C dependency |
| **hayro-jpeg2000** | 0.3.4 | ? | Yes | No | Yes | Yes | ? | ? | Decoder only |

### Recommended Strategy

**Option A (Recommended): Use justjp2 as dependency**
- Already MIT/Apache-2.0 — license compatible
- Pure Rust, 5K LOC, both encode and decode
- Supports lossy (9/7 DWT + ICT) and lossless (5/3 DWT + RCT)
- Multi-tile, ROI cropping, format auto-detection
- Only depends on bitflags + rayon (rayon optional)
- Likely WASM-compatible (pure Rust, no C deps)

**Option B: Fork justjp2 into rasmcore-jp2**
- If we need deeper integration or modifications
- Maintain our own fork with rasmcore conventions
- Add WASM SIMD optimization

**Option C: Use dicom-toolkit-jpeg2000 for decode, justjp2 for encode**
- dicom-toolkit has better test coverage (20K+ images)
- But doesn't seem to support encoding

**Option D: Implement from scratch**
- NOT recommended — JPEG 2000 is significantly more complex than JPEG or VP8
- EBCOT alone is ~2K LOC, DWT ~1K, codestream parsing ~1K
- justjp2 already did this work, MIT/Apache-2.0

### Recommendation: Option A (justjp2 as dependency)
Minimizes effort, gets us to feature parity fastest. Can fork later if needed.

---

## 2. JPEG 2000 Technical Architecture

### Encoding Pipeline (ISO/IEC 15444-1)

```
Input Pixels (RGB/Gray)
    │
    ▼
┌─────────────────┐
│  Color Transform │  ICT (lossy, BT.601-like) or RCT (lossless, integer)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Tiling          │  Partition into tiles (optional, reduces memory)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Forward DWT     │  9/7 wavelet (lossy) or 5/3 wavelet (lossless)
│  (multi-level)   │  Lifting scheme implementation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Quantization    │  Dead-zone scalar (lossy) or identity (lossless)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EBCOT Tier-1    │  Bit-plane coding with MQ arithmetic coder
│  (per code block)│  3 passes: significance, refinement, cleanup
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  EBCOT Tier-2    │  Packet assembly, layer formation
│  + Rate Alloc    │  PCRD optimization (Lagrangian)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Codestream      │  SOC, SIZ, COD, QCD, SOT, SOD, EOC markers
│  Assembly        │  JP2 container wrapping (ISOBMFF boxes)
└─────────────────┘
```

### DWT Details

**CDF 9/7 (Lossy) — Lifting Scheme:**
```
Step 1 (predict): d[n] += α * (s[n] + s[n+1])
Step 2 (update):  s[n] += β * (d[n-1] + d[n])
Step 3 (predict): d[n] += γ * (s[n] + s[n+1])
Step 4 (update):  s[n] += δ * (d[n-1] + d[n])
Step 5 (scale):   s[n] *= K; d[n] /= K

Lifting coefficients:
  α = -1.586134342
  β = -0.052980118
  γ = 0.882911076
  δ = 0.443506852
  K = 1.230174105
```

**Le Gall 5/3 (Lossless) — Integer Lifting:**
```
d[n] = d[n] - floor((s[n] + s[n+1]) / 2)    // predict
s[n] = s[n] + floor((d[n-1] + d[n] + 2) / 4) // update
```

### EBCOT

**Tier-1 (per code block):**
- Bit planes scanned MSB to LSB
- Three coding passes per bit plane:
  1. Significance Propagation: encode newly significant coefficients
  2. Magnitude Refinement: refine already-significant coefficients
  3. Cleanup: encode remaining with run-length optimization
- MQ arithmetic coder with 18 contexts (9 for significance, 5 for sign, 3 for refinement, 1 for run-length)

**Tier-2 (rate allocation):**
- PCRD (Post-Compression Rate-Distortion) optimization
- Lagrangian: minimize Σ D_i subject to Σ R_i ≤ R_target
- Find optimal truncation point for each code block
- Assemble into quality layers

### MQ Arithmetic Coder

- Binary arithmetic coder (same family as JPEG's QM-coder)
- 47 states in probability estimation table
- Conditional exchange for MPS/LPS
- Byte-out with bit-stuffing (0xFF followed by 0-bit pad)
- Reusable with rasmcore-jpeg's arith.rs (similar but different state tables)

### Codestream Markers

| Marker | Code | Description |
|--------|------|-------------|
| SOC | 0xFF4F | Start of codestream |
| SIZ | 0xFF51 | Image and tile size |
| COD | 0xFF52 | Coding style default |
| COC | 0xFF53 | Coding style component |
| QCD | 0xFF5C | Quantization default |
| QCC | 0xFF5D | Quantization component |
| RGN | 0xFF5E | Region of interest |
| POC | 0xFF5F | Progression order change |
| SOT | 0xFF90 | Start of tile-part |
| SOP | 0xFF91 | Start of packet (optional) |
| EPH | 0xFF92 | End of packet header (optional) |
| SOD | 0xFF93 | Start of data |
| EOC | 0xFFD9 | End of codestream |

### JP2 Container (ISOBMFF Boxes)

```
JP2 Signature Box (jP)
File Type Box (ftyp) — brand "jp2 "
JP2 Header Box (jp2h):
  Image Header Box (ihdr) — dimensions, components, bit depth
  Color Specification Box (colr) — ICC profile or enumerated CS
  [Channel Definition Box (cdef)] — optional
  [Resolution Box (res)] — optional
Contiguous Codestream Box (jp2c) — raw J2K codestream
[XML Box (xml)] — optional metadata
[UUID Box (uuid)] — optional extension data
```

---

## 3. Patent Status

- **Part 1 core:** All contributors agreed to royalty-free licensing for core codec
- **Submarine patent risk:** Acknowledged post-2004 but no enforcement to date
- **Part 2 extensions:** Some features may have patent encumbrances
- **Practical status:** Widely implemented (OpenJPEG, Kakadu, etc.) without patent issues
- **Recommendation:** Implement Part 1 only — this covers all common use cases

---

## 4. Integration Plan

### With justjp2 as dependency:

```
crates/rasmcore-image/Cargo.toml:
  justjp2 = "0.1"  # or specific version

domain/decoder/mod.rs:
  - Add JP2/J2K format detection
  - Decode via justjp2::decode()
  - Map output to DecodedImage

domain/encoder/jp2.rs:
  - Jp2EncodeConfig { quality, lossless, tile_size }
  - Encode via justjp2::encode()

WIT interface:
  - jp2-encode-config { quality, lossless, tile-size }
```

### Format detection:
- JP2 container: starts with `\x00\x00\x00\x0C\x6A\x50\x20\x20` (JP2 signature)
- J2K codestream: starts with `\xFF\x4F` (SOC marker)

---

## 5. Complexity Estimates

| Approach | Effort | Risk | Notes |
|----------|--------|------|-------|
| **justjp2 integration** | Low (1 track) | Low | Already MIT/Apache-2.0, pure Rust |
| **Fork + customize** | Medium (2 tracks) | Low | If deeper integration needed |
| **From scratch** | Very High (8+ tracks) | Medium | EBCOT alone is ~2K LOC |

**Recommendation:** Start with justjp2 integration. Fork only if we discover
limitations that can't be addressed upstream.

---

## 6. WASM Feasibility

justjp2 depends on:
- `bitflags` (pure Rust, WASM-safe)
- `rayon` (optional, disable for WASM — single-threaded fallback)

If rayon is optional/feature-gated, justjp2 should compile to wasm32-wasip2
without issues. Need to verify with actual build test.

dicom-toolkit-jpeg2000 has `no_std` support, which is a strong WASM indicator.
