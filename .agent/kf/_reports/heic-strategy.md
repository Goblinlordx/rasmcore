# HEIC/HEIF Support Strategy — Research Report

**Track:** heic-research_20260328082300Z
**Date:** 2026-03-28
**Status:** Research Complete

---

## 1. Patent Landscape

### 1.1 HEVC Patent Pools

HEVC (H.265) has the most fragmented patent landscape of any video codec. Three
pools have administered HEVC patents:

| Pool | Status | Patents | Licensing Model |
|------|--------|---------|-----------------|
| **MPEG LA** | Active | ~2,700 pool-exclusive | Per-device: $0.20/unit after first 100K, $25M annual cap |
| **Access Advance** (fka HEVC Advance) | Active | ~12,400 pool-exclusive | Per-device + content distribution (VDP Pool) |
| **Velos Media** | Dissolved (~2023) | Returned to owners | N/A — patents reverted to individual holders |

**Critical gap:** Only ~35% of HEVC standard-essential patents (SEPs) are covered
by the pools. The remaining ~65% are held by individual companies outside any
pool, creating unquantifiable licensing risk even for pool licensees.

Overlap between pools: ~2,590 patents are licensable through both MPEG LA and
Access Advance, requiring licensees to potentially pay both pools for the same
patents.

### 1.2 Patent Expiry Timeline

- **HEVC standard finalized:** 2013
- **20-year patent term from earliest filing:** Most core patents filed 2010–2013
- **Expected bulk expiry:** ~2030–2033
- **Full unencumbered date:** Approximately November 2030 for the earliest patents,
  but continuation patents and later amendments may extend coverage to ~2035
- **H.264 comparison:** H.264 (2003) is still not fully patent-free as of 2026

There is no firm "royalty-free date" — patents expire individually, and holders
have filed continuations that extend timelines.

### 1.3 Decode-Only Risk Assessment

**Decode is NOT exempt from licensing.** Both MPEG LA and Access Advance license
decoders and encoders alike. Access Advance explicitly covers "HEVC Decoders in
Consumer Products and Cloud-Based Services."

| Risk Factor | Assessment |
|-------------|------------|
| Decode-only exemption | **None** — no pool offers decode-only relief |
| Open source exemption | **None** — patent obligations attach to users/distributors |
| Non-commercial exemption | **Partial** — MPEG LA first 100K units free, but Access Advance has no such threshold |
| Out-of-pool patent risk | **High** — 65% of SEPs are outside pools |
| Enforcement trend | **Active** — Dell/HP disabled HW HEVC decode in late 2025 due to rising royalty costs |

**Conclusion:** Any software that decodes HEVC bitstreams — whether it implements
the decoder or delegates to a system library — exposes the distributor to patent
claims. The risk is real and actively enforced.

---

## 2. Technical Evaluation

### 2.1 Option A: libheif-rs (C FFI Wrapper)

**What:** Rust bindings (`libheif-rs` / `libheif-sys`) wrapping the C `libheif`
library from strukturag. Full-featured HEIC/HEIF/AVIF decode and encode.

| Criterion | Assessment |
|-----------|------------|
| Maturity | High — libheif is production-grade, used by GIMP, ImageMagick |
| HEIC decode | Full support (HEVC via bundled or system codec) |
| AVIF support | Yes (via libdav1d or libaom) |
| Pure Rust | **No** — requires C/C++ libheif + codec libraries |
| WASM (wasm32-wasip2) | **Not compatible** — C FFI dependency, no WASI port |
| Security | C/C++ parser attack surface — contradicts rasmcore's security model |

**Verdict: Rejected.** C FFI dependency makes this incompatible with rasmcore's
pure-Rust, WASM-first architecture. Cannot compile to wasm32-wasip2.

### 2.2 Option B: Pure Rust HEIF Container Parser + Sidecar HEVC Decoder

**What:** Parse the HEIF/ISOBMFF container in pure Rust, extract the HEVC NAL
units, and delegate HEVC decoding to a sidecar component.

#### Container Parsing Options

| Crate | Type | HEIF Support | Maturity | WASM |
|-------|------|-------------|----------|------|
| `heif-rs` (A-K-O-R-A) | Pure Rust HEIF parser | Box-level parsing only | WIP, early stage | Yes |
| `mp4parse` (Mozilla) | Pure Rust ISOBMFF parser | `mif1` brand support | Production (Firefox) | Yes |
| `isobmff` | Pure Rust ISOBMFF parser | Generic box parsing | Low activity | Yes |
| `nom-exif` | Pure Rust metadata parser | HEIF metadata extraction | Active | Yes |

**Best candidate: `mp4parse`** — Mozilla-backed, production-tested in Firefox,
already handles the `mif1` (HEIF image) brand, and is pure Rust with no C
dependencies. However, it's designed for MP4 video metadata, not image extraction.
Would need adaptation or a custom HEIF-specific layer built on top.

**Alternative: custom HEIF parser** — HEIF is a profile of ISOBMFF. The box
structure is well-documented (ISO 14496-12 + ISO 23008-12). A purpose-built parser
for rasmcore's needs (single image, grid image, Exif/XMP metadata) would be
~1–2K lines of Rust and avoids carrying MP4 video parsing baggage.

#### HEVC Decoder Options

| Option | Type | Status | WASM |
|--------|------|--------|------|
| `scuffle-h265` | Pure Rust, header-only | SPS parsing only — not a frame decoder | Yes |
| System libde265/ffmpeg | C FFI | Production | No |
| Custom pure Rust HEVC decoder | Theoretical | Does not exist | — |

**There is no pure Rust HEVC frame decoder.** `scuffle-h265` only parses NAL unit
headers (SPS), not actual frame data. A full HEVC decoder is ~50–100K lines of
highly optimized code — building one from scratch is not feasible.

### 2.3 Option C: AVIF as Royalty-Free Alternative

| Criterion | HEIC (HEVC) | AVIF (AV1) |
|-----------|-------------|------------|
| Container | HEIF (ISOBMFF) | HEIF (ISOBMFF) — same container |
| Codec | HEVC — patented | AV1 — royalty-free |
| Browser support | 12% (Safari only) | 93%+ (Chrome, Firefox, Edge, Safari) |
| Compression | Excellent | Comparable or better |
| Apple ecosystem | Default camera format | Supported since iOS 16 |
| Rust decoder | None (pure Rust) | `rav1d` (ISRG-backed, production) |
| Rust encoder | None | `rav1e` (production) |

rasmcore already supports AVIF via `rav1d` (decode) and `rav1e` (encode). AVIF
uses the same HEIF container format — the only difference is the codec layer.

### 2.4 WASM Compatibility Summary

| Component | WASM Compatible | Notes |
|-----------|----------------|-------|
| HEIF container parser (pure Rust) | Yes | Custom or mp4parse-based |
| HEVC decoder (any) | No | No pure Rust decoder exists |
| HEVC sidecar (host-provided) | Yes | Via WIT import interface |
| AVIF decode (rav1d) | Yes | Already in rasmcore |
| AVIF encode (rav1e) | Yes | Already in rasmcore |

---

## 3. Recommended Strategy

### 3.1 Architecture: Sidecar Plugin Pattern

Implement HEIC support using rasmcore's sidecar plugin pattern:

```
┌─────────────────────────────────────────────────┐
│  rasmcore-heif (core module, pure Rust, WASM)   │
│                                                 │
│  ┌───────────────┐    ┌──────────────────────┐  │
│  │ HEIF Container │    │  Codec Dispatch      │  │
│  │ Parser         │───▶│                      │  │
│  │ (ISOBMFF boxes)│    │  HEVC → sidecar WIT  │  │
│  │                │    │  AV1  → rav1d (core) │  │
│  └───────────────┘    │  JPEG → core decoder  │  │
│                        └──────────────────────┘  │
└──────────────────────────┬───────────────────────┘
                           │ WIT import (decode-hevc)
                           ▼
┌─────────────────────────────────────────────────┐
│  rasmcore-hevc-sidecar (separate distribution)  │
│                                                 │
│  Host-provided HEVC decoder                     │
│  (user supplies, not bundled with rasmcore)      │
│  Options: system libde265, ffmpeg, HW decoder   │
└─────────────────────────────────────────────────┘
```

### 3.2 Implementation Path

**Phase 1 — HEIF Container (rasmcore-heif crate, core module)**
- Pure Rust HEIF/ISOBMFF parser (~1–2K lines)
- Parse `ftyp`, `meta`, `iloc`, `iprp`, `iref`, `idat` boxes
- Extract codec type, image dimensions, grid layout, Exif/XMP
- Extract raw codec bitstream (NAL units for HEVC, OBUs for AV1)
- Dispatch to appropriate decoder based on codec FourCC:
  - `hvc1`/`hev1` → sidecar HEVC interface (WIT import)
  - `av01` → internal rav1d decoder
  - `jpeg` → internal JPEG decoder
- **No patent risk** — container parsing is royalty-free

**Phase 2 — Sidecar Interface (WIT definition)**
- Define `decode-hevc` WIT import interface
- Input: raw HEVC NAL units + SPS/PPS
- Output: decoded pixel buffer (RGBA/YUV)
- Host provides the implementation — rasmcore never bundles HEVC code
- **No patent risk for rasmcore** — patent obligation transfers to the host

**Phase 3 — Reference Sidecar (separate repo/distribution)**
- Thin wrapper around system `libde265` or FFmpeg's HEVC decoder
- Distributed separately from rasmcore (different license, different repo)
- Users who need HEIC decode opt-in by providing this sidecar
- Clear documentation that users assume patent licensing responsibility

### 3.3 What rasmcore Ships vs. What Users Provide

| Component | Ships with rasmcore | User provides |
|-----------|:-------------------:|:-------------:|
| HEIF container parser | Yes | — |
| AVIF decode (AV1) | Yes | — |
| HEVC decode | — | Yes (sidecar) |
| HEVC encode | — | No (out of scope) |

### 3.4 AVIF-as-Alternative Strategy

For users who don't need HEIC specifically:
- Recommend AVIF as the royalty-free alternative for all new content
- rasmcore already has full AVIF encode/decode
- Same HEIF container — shared parser code
- 93%+ browser support vs HEIC's 12%
- Document HEIC→AVIF transcoding workflow (requires sidecar for decode step)

---

## 4. Risk Summary

| Risk | Mitigation |
|------|------------|
| Patent liability from HEVC decode | Sidecar pattern — rasmcore never contains HEVC code |
| No pure Rust HEVC decoder | Sidecar delegates to host-provided C decoder |
| HEIF container complexity | Purpose-built parser scoped to image use case |
| Sidecar UX friction | Clear docs, reference implementation, graceful error when sidecar absent |
| WASM compatibility | Container parser is pure Rust; sidecar uses WIT imports |

---

## 5. Minimum Viable Implementation

The smallest useful deliverable is:

1. **HEIF container parser** that extracts image data and metadata
2. **WIT sidecar interface** definition for HEVC decode
3. **Graceful fallback** — when no HEVC sidecar is available, return structured
   error with metadata (dimensions, codec, Exif) but no pixel data
4. **AVIF pass-through** — HEIF files using AV1 codec decoded natively

This gives rasmcore the ability to handle the HEIF container format universally,
decode AVIF-in-HEIF natively, and decode HEIC when the user provides an HEVC
sidecar — all without any patent risk to the rasmcore project.

---

## Sources

- [Access Advance — HEVC Advance Patent Pool](https://accessadvance.com/licensing-programs/hevc-advance/)
- [MPEG LA — HEVC Patent Portfolio](https://www.mpegla.com/wp-content/uploads/HEVCweb.pdf)
- [GreyB — Only 35% HEVC SEPs Covered by Patent Pools](https://greyb.com/resources/reports/hevc-patent-landscape-analysis/)
- [Streaming Learning Center — HEVC IP Mess](https://streaminglearningcenter.com/codecs/hevc-ip-mess-worse-think.html)
- [Tom's Hardware — Dell/HP Disable HW HEVC Decode](https://www.tomshardware.com/pc-components/gpus/dell-and-hp-disable-hardware-h-265-decoding-on-select-pcs-due-to-rising-royalty-costs-companies-could-save-big-on-hevc-royalties-but-at-the-expense-of-users)
- [Wikipedia — HEVC](https://en.wikipedia.org/wiki/High_Efficiency_Video_Coding)
- [Wikipedia — HEIF](https://en.wikipedia.org/wiki/High_Efficiency_Image_File_Format)
- [GitHub — heif-rs (A-K-O-R-A)](https://github.com/A-K-O-R-A/heif-rs)
- [GitHub — mp4parse-rust (Mozilla)](https://github.com/mozilla/mp4parse-rust)
- [GitHub — libheif (strukturag)](https://github.com/strukturag/libheif)
- [crates.io — scuffle-h265](https://crates.io/crates/scuffle-h265)
- [crates.io — libheif-rs](https://crates.io/crates/libheif-rs)
- [Cloudinary — Advanced Image Formats](https://cloudinary.com/blog/advanced-image-formats-and-when-to-use-them)
- [DEV Community — AVIF vs WebP vs HEIC vs JPEG XL 2026](https://dev.to/serhii_kalyna_730b636889c/avif-vs-webp-vs-heic-vs-jpeg-xl-which-image-format-should-you-use-in-2026-4gn0)
