# Codec Parity Audit — rasmcore vs ImageMagick vs libvips

**Date:** 2026-03-28
**Track:** codec-audit-patents_20260328052209Z

---

## 1. Format Support Matrix

### Legend

| Symbol | Meaning |
|--------|---------|
| D | Decode supported |
| E | Encode supported |
| P | Pipeline write sink |
| W | WIT interface exposed |
| I | ICC profile support |
| M | Metadata read |
| - | Not supported |
| * | Feature-gated in image crate (trivial to enable) |

### rasmcore Current State

| Format | Decode | Encode | Pipeline | WIT | ICC | Metadata |
|--------|--------|--------|----------|-----|-----|----------|
| **JPEG** | D | E (zenjpeg) | P | W | I (extract+embed) | M (EXIF 7 fields) |
| **PNG** | D | E (image) | P | W | I (extract+embed) | - |
| **WebP** | D | E (in-progress) | P | W | - | - |
| **GIF** | D | E (image) | P | W | - | - |
| **TIFF** | D | - | - | - | - | - |
| **BMP** | D | - | - | - | - | - |
| **AVIF** | D | - | - | - | - | - |
| **ICO** | D | - | - | - | - | - |
| **QOI** | D | - | - | - | - | - |
| PNM/PPM | -* | -* | - | - | - | - |
| TGA | -* | -* | - | - | - | - |
| EXR | -* | -* | - | - | - | - |
| HDR | -* | -* | - | - | - | - |
| DDS | -* | -* | - | - | - | - |
| JPEG 2000 | - | - | - | - | - | - |
| JPEG XL | - | - | - | - | - | - |
| HEIF/HEIC | - | - | - | - | - | - |
| SVG | - | N/A | - | - | - | - |
| RAW | - | N/A | - | - | - | - |

### Reference Implementation Comparison

| Format | rasmcore R/W | ImageMagick R/W | libvips R/W | Gap |
|--------|-------------|-----------------|-------------|-----|
| JPEG | R/W | R/W | R/W | **none** |
| PNG | R/W | R/W | R/W | **none** |
| GIF | R/W | R/W | R/W | **none** |
| WebP | R/W* | R/W | R/W | **in-progress** |
| TIFF | R/- | R/W | R/W | **encode missing** |
| BMP | R/- | R/W | R/W (magick) | encode missing |
| AVIF | R/- | R/W | R/W | **encode missing** |
| ICO | R/- | R/W | R/W (magick) | encode missing |
| QOI | R/- | R/W | -/- | encode missing |
| HEIF/HEIC | -/- | R/W | R/W | **full gap (patent)** |
| JPEG XL | -/- | R/W | R/W | **full gap** |
| JPEG 2000 | -/- | R/W | R/W | full gap |
| EXR | -/- | R/W | R/- | full gap |
| HDR | -/- | R/W | R/W | full gap |
| PNM/PPM | -/- | R/W | R/W | full gap |
| TGA | -/- | R/W | R/W (magick) | full gap |
| DDS | -/- | R/W | -/- | full gap |
| SVG | -/- | R/- | R/- | rasterize gap |
| RAW | -/- | R/- | R/- | decode gap |

---

## 2. Patent and License Assessment

### Classification Key

| Status | Meaning |
|--------|---------|
| CLEAR | No patents, free to implement |
| EXPIRED | Had patents, now expired |
| ROYALTY-FREE | Patents exist but RF license available |
| ENCUMBERED | Active patents requiring license fees |

### Per-Format Patent Status

| Format | Patent Status | Details |
|--------|-------------|---------|
| **JPEG** | EXPIRED | Forgent patent invalidated 2006, expired 2006. Princeton patent expired 2007. |
| **PNG** | CLEAR | Designed patent-free (response to GIF/LZW controversy). |
| **GIF** | EXPIRED | Unisys LZW patent expired June 2003. International by July 2004. |
| **WebP** | ROYALTY-FREE | Google irrevocable RF patent grant on VP8. Reciprocal cross-license. |
| **TIFF** | CLEAR | Container never patented. LZW compression patent expired 2003. |
| **BMP** | CLEAR | Never patented. |
| **AVIF** | ROYALTY-FREE | AOM Patent License 1.0 (RF). Residual risk: Sisvel patent pool (key patent EP 2627085 revoked May 2025). |
| **HEIF/HEIC** | **ENCUMBERED** | Three HEVC patent pools: MPEG-LA ($0.20/device), HEVC Advance ($2.03/device), Velos Media. Nokia actively litigating (Acer/ASUS Germany ban Jan 2026). |
| **JPEG XL** | ROYALTY-FREE | ISO/IEC 18181. Google/Cloudinary RF patent grant. Apache-2.0 reference impl. |
| **JPEG 2000** | EXPIRED | Core patents expired (~2017-2020). Historical ambiguity now moot. |
| **QOI** | CLEAR | CC0 spec (public domain) + MIT reference implementation. |
| **EXR** | CLEAR | BSD-3-Clause. Academy Software Foundation project. |
| **HDR** | CLEAR | Permissive license since 2002. No patents. |
| **SVG** | ROYALTY-FREE | W3C Royalty-Free Patent Policy. |
| **ICO** | CLEAR | Never patented. Simple BMP/PNG container. |
| **PNM/PPM** | CLEAR | Trivial format, no patents possible. |
| **TGA** | CLEAR | Non-proprietary format. |
| **DDS** | EXPIRED | S3TC patent expired Oct 2017. BC1-BC3 free. |
| **RAW** | CLEAR | Bayer patent long expired. Avoid vendor-specific advanced demosaicing algorithms. |

### HEIF/HEIC Sidecar Strategy

HEIF/HEIC is the **only** format requiring patent licensing. The product.yaml already anticipates this:

> "Patent-encumbered codec support via sidecar plugin distribution"

Recommended approach:
- HEIF/HEIC codec compiled as a **separate WASM component** (sidecar)
- Distributed under a different license/package
- Loaded dynamically by the host application
- Users who need HEIC support accept the patent licensing terms
- Core rasmcore stays patent-clean

**Note:** No pure-Rust HEIF crate exists. `libheif-rs` wraps C++ and is not WASM-compatible. HEIF support would be native-only via sidecar.

---

## 3. Rust Crate Evaluation

### Tier 1: Image Crate Feature Flags (trivial, zero new dependencies)

These formats are already supported by the `image` crate. Just enable the feature flag and add to decoder/encoder dispatch.

| Format | Feature | Decode | Encode | License | WASM |
|--------|---------|--------|--------|---------|------|
| PNM/PPM | `pnm` | Yes | Yes (PnmEncoder) | MIT/Apache-2.0 | Yes |
| TGA | `tga` | Yes | No | MIT/Apache-2.0 | Yes |
| HDR | `hdr` | Yes | No | MIT/Apache-2.0 | Yes |
| DDS | `dds` | Yes | No | MIT/Apache-2.0 | Yes |
| QOI encode | `qoi` (already enabled) | Yes | Yes | MIT/Apache-2.0 | Yes |
| BMP encode | `bmp` (already enabled) | Yes | Yes | MIT/Apache-2.0 | Yes |
| ICO encode | `ico` (already enabled) | Yes | Yes | MIT/Apache-2.0 | Yes |
| TIFF encode | `tiff` (already enabled) | Yes | Yes | MIT/Apache-2.0 | Yes |

### Tier 2: External Pure-Rust Crates (moderate effort, WASM-compatible)

| Format | Crate | License | Pure Rust | WASM | Maturity | Notes |
|--------|-------|---------|-----------|------|----------|-------|
| **AVIF encode** | `ravif` | BSD-3 | Yes (rav1e) | Partial | High (2.3M dl/mo) | WASM encode very slow without SIMD |
| **JXL decode** | `jxl-oxide` | MIT/Apache-2.0 | Yes | Yes | High (48K dl/mo) | Excellent decoder |
| **SVG rasterize** | `resvg` | Apache-2.0 | Yes | Yes | High (actively maintained) | Already in tech-stack.yaml |
| **EXR** | `exr` | BSD-3 | Yes | Yes | High (40M total dl) | Used by image crate internally |
| **Camera RAW** | `rawloader` | **LGPL-2.1** | Yes | Yes | Moderate | LGPL requires care. rawloader-wasm fork exists. |
| **EXIF read/write** | `little_exif` | MIT/Apache-2.0 | Yes | Yes | Moderate | Only pure-Rust EXIF writer. Supports JPEG/PNG/TIFF/WebP/HEIF/AVIF/JXL. |
| **XMP write** | `xmp-writer` | MIT/Apache-2.0 | Yes | Yes | Moderate | Write-only. XMP read via quick-xml. |
| **PNG chunks** | `png` (image-rs) | MIT/Apache-2.0 | Yes | Yes | High | tEXt/zTXt/iTXt built-in. |

### Tier 3: License-Restricted or Native-Only

| Format | Crate | License | Pure Rust | WASM | Issue |
|--------|-------|---------|-----------|------|-------|
| **JXL encode** | `jxl-encoder` | **AGPL-3.0** | Yes | Yes | AGPL incompatible with proprietary use |
| **JXL encode** | `jpegxl-rs` | BSD-3 | No (C++) | No | Not WASM-compatible |
| **HEIF/HEIC** | `libheif-rs` | MIT/LGPL | No (C++) | No | No pure-Rust option exists |
| **JPEG 2000** | `openjp2` | BSD-2 | Rust port | Unconfirmed | Low-quality mechanical port |
| **IPTC** | `rexiv2` | **GPL-3.0** | No (C++) | No | GPL + C++ deps |
| **XMP full** | `xmp_toolkit` | MIT/Apache-2.0 | No (C++) | No | Adobe C++ SDK wrapper |

### Key Risks

1. **HEIF/HEIC** — No pure-Rust implementation. Completely blocked for WASM. Biggest ecosystem gap.
2. **JXL encode** — Only pure-Rust encoder is AGPL-3.0. Monitor for permissive alternatives.
3. **AVIF encode in WASM** — rav1e WASM support is experimental. Very slow without hardware SIMD.
4. **IPTC** — No pure-Rust crate. Simple enough to write a minimal parser.
5. **JPEG 2000** — Fragmented ecosystem, no quality pure-Rust option. Low priority.

---

## 4. Metadata Support Audit

### Metadata Types Per Format

| Format | EXIF | XMP | IPTC | ICC | Format-Specific |
|--------|------|-----|------|-----|-----------------|
| **JPEG** | Yes (APP1) | Yes (APP1 XMP) | Yes (APP13) | Yes (APP2) | JFIF (APP0) |
| **PNG** | Yes (eXIf chunk) | Yes (iTXt) | No | Yes (iCCP) | tEXt, zTXt, iTXt |
| **WebP** | Yes (EXIF chunk) | Yes (XMP chunk) | No | Yes (ICCP chunk) | VP8/VP8L headers |
| **TIFF** | Yes (IFD tags) | Yes (tag 700) | Yes (tag 33723) | Yes (tag 34675) | GeoTIFF, SubIFD |
| **GIF** | No | No | No | No | Comment extension (0xFE) |
| **AVIF** | Yes (Exif box) | Yes (XMP box) | No | Yes (colr box) | HEIF container metadata |
| **HEIF/HEIC** | Yes (Exif box) | Yes (XMP box) | No | Yes (colr box) | HEIF container metadata |
| **JPEG XL** | Yes (Exif box) | Yes (XMP box) | No | Yes (ICC box) | JXL-specific boxes |
| **BMP** | No | No | No | No | Info header only |
| **ICO** | No | No | No | No | Directory entries |
| **QOI** | No | No | No | No | 14-byte header only |
| **EXR** | No | No | No | No | Named attributes (custom key-value) |
| **PNM** | No | No | No | No | ASCII header comment |

### Current rasmcore Metadata Support

| Capability | Status | Implementation |
|-----------|--------|----------------|
| EXIF read | Partial | kamadak-exif: 7 fields (orientation, dims, camera, date, software) |
| EXIF write | None | — |
| XMP read | None | — |
| XMP write | None | — |
| IPTC read | None | — |
| IPTC write | None | — |
| ICC extract | JPEG + PNG | Manual APP2/iCCP parsing |
| ICC embed | JPEG + PNG | Manual APP2/iCCP injection |
| ICC transform | Yes | moxcms: ICC-to-sRGB conversion |
| PNG text chunks | None | — |
| GIF comments | None | — |
| Auto-orient | Yes | EXIF orientation tag (1-8) applied |

### Recommended Crates for Metadata Gaps

| Capability | Crate | License | Pure Rust | WASM | Notes |
|-----------|-------|---------|-----------|------|-------|
| EXIF read/write | `little_exif` | MIT/Apache-2.0 | Yes | Yes | Replaces kamadak-exif for write; supports 7 formats |
| XMP write | `xmp-writer` | MIT/Apache-2.0 | Yes | Yes | Serialize XMP to bytes |
| XMP read | `quick-xml` | MIT | Yes | Yes | Parse XMP (which is XML) |
| PNG chunks | `png` crate | MIT/Apache-2.0 | Yes | Yes | tEXt/zTXt/iTXt built-in |
| IPTC | Custom parser | — | Yes | Yes | Simple binary format, ~200 LOC |

---

## 5. Metadata API Design

### Design Principles

1. **Unified domain type** — `MetadataSet` carrying all metadata kinds (EXIF, XMP, IPTC, ICC, format-specific)
2. **Passthrough by default** — metadata flows unchanged from source to sink unless transformed
3. **Host-side mapping** — transform logic runs on the host (TS, Go, Python), not in WASM
4. **Streaming read** — parse container headers to extract metadata without decoding pixels
5. **Pipeline write** — metadata attached to pipeline, embedded during encode sink phase

### Three-Tier Execution Model

| Tier | When | Strategy | Performance |
|------|------|----------|-------------|
| Same-format passthrough | JPEG-to-JPEG, PNG-to-PNG | Raw byte splice, no parsing | Fastest (zero-copy) |
| Cross-format passthrough | JPEG-to-PNG, TIFF-to-WebP | Parse to MetadataSet, re-serialize | Moderate |
| Transform | Any with user mapping | Parse, map (host-side), re-serialize | Moderate + host overhead |

### WIT Interface Sketch

```wit
/// Unified metadata container — opaque to the pipeline, structured for the host
record metadata-set {
    exif: option<buffer>,     // Raw EXIF bytes (parseable by host SDK)
    xmp: option<buffer>,      // Raw XMP bytes (XML, parseable by host)
    iptc: option<buffer>,     // Raw IPTC-IIM bytes
    icc-profile: option<buffer>,  // ICC profile bytes
    format-specific: list<metadata-chunk>,  // PNG text, GIF comment, etc.
}

record metadata-chunk {
    key: string,
    value: buffer,
}

/// Read metadata from encoded data WITHOUT decoding pixels
read-metadata: func(data: buffer) -> result<metadata-set, rasmcore-error>;

/// Write sinks accept optional metadata to embed
/// (extends existing write-jpeg, write-png, etc.)
write-jpeg: func(source: node-id, config: jpeg-write-config, metadata: option<metadata-set>)
    -> result<buffer, rasmcore-error>;
```

### SDK Integration Pattern

```typescript
// TypeScript SDK — metadata flows through host
const meta = rasmcore.metadata.read(inputBytes);  // WASM: extract

const mapped = meta
  .exclude("gps", "thumbnail")          // Host: filter
  .include("icc", "exif.orientation")    // Host: allowlist
  .set("copyright", "2026 Acme Inc");   // Host: inject

const output = pipeline
  .read(inputBytes)
  .resize(800, 600)
  .writeJpeg({ quality: 85 }, mapped);   // WASM: encode + embed
```

---

## 6. Implementation Roadmap

### Priority Tiers

#### Tier 1 — Trivial (image crate feature flags + encoder modules)

No new dependencies. Follow existing per-format encoder pattern.

| Format | Work | Effort | Dependencies |
|--------|------|--------|--------------|
| TIFF encode | New encoder module + WIT + pipeline sink | 1 day | image crate (already enabled) |
| BMP encode | New encoder module + WIT + pipeline sink | 0.5 day | image crate (already enabled) |
| ICO encode | New encoder module + WIT + pipeline sink | 0.5 day | image crate (already enabled) |
| QOI encode | New encoder module + WIT + pipeline sink | 0.5 day | image crate (already enabled) |
| PNM decode+encode | Enable feature, add to dispatch | 0.5 day | Enable `pnm` feature |
| TGA decode | Enable feature, add to dispatch | 0.25 day | Enable `tga` feature |
| HDR decode | Enable feature, add to dispatch | 0.25 day | Enable `hdr` feature |
| DDS decode | Enable feature, add to dispatch | 0.25 day | Enable `dds` feature |
| EXR decode+encode | Enable feature, add to dispatch | 0.5 day | Enable `exr` feature |

**Estimated total: 1-2 tracks, 3-4 days**

#### Tier 2 — Moderate (external pure-Rust crates)

| Format | Work | Effort | Crate | License |
|--------|------|--------|-------|---------|
| AVIF encode | New encoder module, integrate ravif | 2-3 days | ravif (BSD-3) | Clean |
| JXL decode | New decoder path, integrate jxl-oxide | 1-2 days | jxl-oxide (MIT/Apache-2.0) | Clean |
| SVG rasterize | New decoder path, integrate resvg | 2-3 days | resvg (Apache-2.0) | Clean |
| RAW decode | New decoder path, integrate rawloader | 2-3 days | rawloader (**LGPL-2.1**) | Needs review |

**Estimated total: 2-4 tracks, 1-2 weeks**

#### Tier 3 — Metadata Unification

| Work | Effort | Crates |
|------|--------|--------|
| MetadataSet domain type + WIT interface | 2 days | — |
| EXIF read/write via little_exif | 1-2 days | little_exif (MIT/Apache-2.0) |
| XMP read (quick-xml) + write (xmp-writer) | 1-2 days | quick-xml + xmp-writer |
| IPTC parser (custom, ~200 LOC) | 0.5 day | — |
| PNG text chunk integration | 0.5 day | png crate (built-in) |
| Streaming metadata read (header-only parsing) | 2-3 days | — |
| Pipeline write sink metadata parameter | 1-2 days | — |

**Estimated total: 2-3 tracks, 1-2 weeks**

#### Tier 4 — Complex / License-Restricted

| Format | Work | Blocker | Strategy |
|--------|------|---------|----------|
| JXL encode | Integrate encoder | AGPL-3.0 license on jxl-encoder | Wait for permissive crate, or native-only via jpegxl-rs |
| HEIF/HEIC | Sidecar plugin | No pure-Rust crate, patent encumbered | Separate WASM component, optional distribution |
| JPEG 2000 | Low-quality Rust port | No mature crate | Low priority, native-only if needed |

**Deferred until ecosystem matures or demand justifies.**

### Recommended Track Sequence

```
1. codec-tier1-formats     — Enable all image crate formats + simple encoders
2. encoder-avif            — AVIF encode via ravif
3. decoder-jxl             — JPEG XL decode via jxl-oxide
4. metadata-unification    — MetadataSet type, streaming read, pipeline write
5. metadata-exif-xmp       — EXIF write + XMP read/write via little_exif + xmp-writer
6. decoder-svg             — SVG rasterize via resvg
7. decoder-raw             — Camera RAW decode via rawloader (LGPL review)
8. codec-heif-sidecar      — HEIF/HEIC sidecar plugin architecture (deferred)
```

---

## 7. Summary

**Current:** 9 decode formats, 4 encode formats (JPEG, PNG, WebP, GIF)
**After Tier 1:** 14 decode formats, 9 encode formats
**After Tier 2:** 17 decode formats, 10 encode formats
**After all tiers:** 18+ decode, 11+ encode — parity with libvips native formats

**Patent clear:** Every format except HEIF/HEIC is safe to implement. AVIF has residual Sisvel risk but AOM RF license covers it. HEIF/HEIC requires sidecar isolation.

**Metadata:** Current ad-hoc approach (EXIF read + ICC extract/embed) should be replaced with unified MetadataSet type. All metadata crates are pure-Rust and WASM-compatible except IPTC (custom parser needed, simple).
