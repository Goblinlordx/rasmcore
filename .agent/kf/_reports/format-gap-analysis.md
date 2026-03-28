# Format Gap Analysis — rasmcore vs ImageMagick vs libvips

**Date:** 2026-03-28
**Goal:** Meet or exceed ImageMagick and libvips format support for raster images.

---

## Current Coverage (15 formats, read+write)

| Format | Decode | Encode | Backend | Native Crate |
|--------|--------|--------|---------|-------------|
| PNG | Yes | Yes | image crate + fdeflate | rasmcore-deflate |
| JPEG | Yes | Yes | rasmcore-jpeg (replacing zenjpeg) | rasmcore-jpeg |
| WebP | Yes | Yes (lossy+lossless) | rasmcore-webp + image-webp | rasmcore-webp |
| GIF | Yes | Yes | image crate | rasmcore-lzw |
| TIFF | Yes | Yes (LZW/Deflate/PackBits) | tiff crate | — |
| AVIF | Yes | Yes | image crate (rav1e) | — |
| BMP | Yes | Yes (palette, RLE, bitfields) | rasmcore-bmp | rasmcore-bmp |
| ICO | Yes | Yes | image crate | — |
| QOI | Yes | Yes | rasmcore-qoi | rasmcore-qoi |
| TGA | Yes | Yes (RLE, palette) | rasmcore-tga | rasmcore-tga |
| HDR/Radiance | Yes | Yes | image crate | — |
| PNM (PPM/PGM/PBM/PAM) | Yes | Yes | rasmcore-pnm | rasmcore-pnm |
| OpenEXR | Yes | Yes | image crate | — |
| DDS | Yes | Yes | rasmcore-image | — |
| JPEG XL | Yes | **Scaffold only** | jxl-oxide (decode) | rasmcore-jxl (planned) |

---

## Gaps vs ImageMagick + libvips

### High Priority — Real user demand, both IM and vips support these

| Format | IM | vips | rasmcore | Gap | Notes |
|--------|-----|------|----------|-----|-------|
| **JPEG XL encode** | RW | RW | R only | Write | 8 tracks planned (rasmcore-jxl), scaffold exists |
| **HEIC/HEIF** | RW | RW | None | RW | Apple's default camera format. HEVC patents complicate pure Rust. See research notes below. |
| **JPEG 2000** | RW | RW | None | RW | Medical imaging (DICOM), digital cinema (DCP), archival. See research notes below. |

### Medium Priority — Useful but narrower audience

| Format | IM | vips | rasmcore | Gap | Notes |
|--------|-----|------|----------|-----|-------|
| **SVG** | RW | R | None | R (render) | Vector format — requires rendering engine. resvg (pure Rust) is mature. Read-only (rasterize SVG to pixels). Writing SVG is a different domain (vector graphics generation). |
| **PDF** | RW | R | None | R (render) | Document format, not raster image. Render-to-pixels via pdf-rs or pdfium. Low overlap with image processing. |
| **FITS** | RW | RW | None | RW | Astronomy format. Niche but important in scientific imaging. Pure Rust crate (fitsio) exists. |

### Low Priority — Legacy or very niche

| Format | IM | vips | rasmcore | Gap | Notes |
|--------|-----|------|----------|-----|-------|
| **PostScript/EPS** | RW | R (via IM) | None | R | Legacy print format. Declining use. Ghostscript dependency in IM/vips. |
| **XPM/XBM** | RW | — | None | RW | Unix icon formats. Trivial to implement if needed. |
| **PCX** | RW | — | None | RW | DOS-era format. Very niche. |
| **SGI/RGB** | RW | — | None | RW | Silicon Graphics format. Historical. |
| **Sun Raster** | RW | — | None | RW | SunOS format. Historical. |
| **WBMP** | RW | — | None | RW | Wireless bitmap. Feature phones era. |

---

## Research Notes for Future Tracks

### HEIC/HEIF
- **Standard:** ISO/IEC 23008-12 (HEIF container) + ISO/IEC 23008-2 (HEVC codec)
- **Patent status:** HEVC has active patent pools (MPEG LA, Access Advance, Velos Media). Commercial use requires licensing.
- **Pure Rust options:**
  - **Decode:** `libheif-rs` wraps C libheif. No pure Rust HEVC decoder exists.
  - **AVIF alternative:** AVIF uses the same HEIF container but with AV1 codec (royalty-free). rasmcore already supports AVIF — recommend AVIF as the royalty-free alternative to HEIC.
- **Recommendation:** Support HEIC decode only (for reading Apple photos), via optional C dependency or by converting HEIC→AVIF at ingest. Do NOT implement HEVC encoder (patent risk). For the sidecar plugin pattern described in product.yaml, HEIC decode is a perfect candidate.
- **Effort:** Medium (decode wrapper) to Very High (native HEVC implementation — not recommended)

### JPEG 2000
- **Standard:** ISO/IEC 15444 (multiple parts)
- **Patent status:** Core codec is royalty-free. Some extensions have patents.
- **Pure Rust options:**
  - `jpeg2000` crate wraps OpenJPEG (C library)
  - `openjp2` is a partial pure Rust port (incomplete)
  - No production-ready pure Rust J2K implementation exists
- **Use cases:** Medical (DICOM mandates J2K), digital cinema (DCP), archival (JPEG 2000 Part 2)
- **Recommendation:** Phase 1: decode via openjpeg wrapper (optional C dep). Phase 2: research pure Rust implementation from ISO spec (similar to VP8/JPEG approach). Phase 3: sidecar plugin for patent-encumbered extensions.
- **Effort:** Very High for pure Rust (wavelet transform + embedded block coding is complex)

### SVG Rendering
- **Pure Rust options:**
  - `resvg` — production-quality SVG renderer, pure Rust, used by Firefox
  - Renders SVG to pixel buffer (rasterization)
  - Does NOT support full SVG spec (no scripting, limited animation)
- **Recommendation:** Integrate resvg as an optional dependency for SVG→raster conversion. This is read-only (SVG in, pixels out). SVG generation is a different domain.
- **Effort:** Low (integration only — resvg does the hard work)

### FITS (Astronomy)
- **Standard:** FITS 4.0 (NASA/IAU standard)
- **Pure Rust options:** `fitsio` crate exists but wraps cfitsio (C). Pure Rust parsers are minimal.
- **Recommendation:** Low priority unless targeting scientific imaging users. Implement later if demand exists.
- **Effort:** Medium

---

## rasmcore Advantages Over IM/vips

| Advantage | Detail |
|-----------|--------|
| **QOI format** | Native support — neither IM nor vips has it |
| **Pure Rust / WASM** | Compiles to wasm32-wasip2, runs in browsers, edge compute |
| **No C dependencies** | Core codec crates are pure Rust (rasmcore-webp, rasmcore-jpeg, etc.) |
| **Security** | No C/C++ parser vulnerabilities (buffer overflows, etc.) |
| **Modular** | Each codec is a standalone reusable crate |
| **Component Model** | WIT interfaces enable cross-language composability |
| **DDS read+write** | Both IM and vips have limited DDS support |

---

## Recommended Track Ordering for Gap Closure

### Already in progress:
1. JPEG encoder (rasmcore-jpeg) — 8 tracks, eliminates AGPL
2. JXL encoder (rasmcore-jxl) — 8+ tracks planned
3. WebP integration — 1 track remaining

### Next priorities (future):
4. **SVG decode via resvg** — 1 track (integration only)
5. **HEIC decode** — 1-2 tracks (sidecar plugin pattern for patent-encumbered codec)
6. **JPEG 2000 decode** — 1-2 tracks (openjpeg wrapper initially)
7. **JPEG 2000 pure Rust** — 6-8 tracks (long-term, from ISO spec)
8. **FITS** — 1-2 tracks (if demand exists)

### After these, rasmcore would support 20+ formats — exceeding both IM and vips for raster image coverage while maintaining pure Rust / WASM portability.
