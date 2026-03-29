# Binary Size Analysis & Native Codec Roadmap

## Current Size (2026-03-29)

| Measure | Size |
|---------|------|
| WASM release | 11.3 MB |
| Stripped | 11.3 MB |
| **Gzipped** | **3.4 MB** |

## Codec Status

### Native (zero external dependency)

| Format | Decode | Encode | Notes |
|--------|--------|--------|-------|
| **JPEG** | ✓ | ✓ | rasmcore-jpeg, mozjpeg-quality trellis |
| **WebP** | ✓ | ✓ | rasmcore-webp, VP8 encoder |
| **BMP** | ✓ | ✓ | rasmcore-bmp (feature-gated) |
| **QOI** | ✓ | ✓ | rasmcore-qoi (feature-gated) |
| **PNM** | ✓ | ✓ | rasmcore-pnm (feature-gated) |
| **FITS** | ✓ | ✓ | rasmcore-fits |
| **DDS** | — | ✓ | Native encode only |

### External dependency (to be replaced with native)

| Format | Decode | Encode | Current Dep | Dep Size | Priority |
|--------|--------|--------|-------------|----------|----------|
| **PNG** | image crate | image crate | image (png) | ~1 MB | **HIGH** |
| **GIF** | image crate | image crate | image (gif) | ~0.5 MB | MEDIUM |
| **TIFF** | image crate | tiff crate | image + tiff | ~2 MB | MEDIUM |
| **ICO** | image crate | image crate | image (ico) | ~0.3 MB | LOW |
| **TGA** | image crate | image crate | image (tga) | ~0.3 MB | LOW |
| **HDR** | image crate | image crate | image (hdr) | ~0.3 MB | LOW |
| **DDS** | image crate | — | image (dds) | ~0.5 MB | LOW |
| **EXR** | image crate | image crate | exr crate | ~5.9 MB | LOW (niche) |
| **AVIF** | image crate | ravif+rav1e | ravif+rav1e | ~9 MB | LOW (AV1 patent) |
| **JXL** | jxl-oxide | — | jxl-oxide | ~2.4 MB | MEDIUM |
| **JP2** | justjp2 | justjp2 | justjp2 | ~1 MB | LOW |
| **SVG** | resvg | N/A | resvg+usvg+rustybuzz | ~12 MB | LOW |
| **HEIF** | native (opt) | — | feature-gated | varies | MEDIUM |

### Transitive bloat (cannot remove directly)

| Dependency | Size | Pulled By | Impact |
|------------|------|-----------|--------|
| rayon | ~4 MB | rav1e, libblur | **Does not work in WASM** — pure dead weight |
| zerocopy | ~15 MB rlib | Multiple (tree-shaken in final binary) | Moderate |
| nom | ~4.3 MB | Format parsers | Moderate |
| rustybuzz+ttf-parser | ~7.4 MB | resvg (SVG text) | Feature-gate SVG to remove |

## Roadmap to Minimal Binary

### Phase 1: Feature-gate heavy optional codecs
- Gate SVG behind `feature = "svg"` (saves ~12 MB)
- Gate AVIF behind `feature = "avif"` (saves ~9 MB, also removes rayon)
- Gate EXR behind `feature = "exr"` (saves ~6 MB)
- **Estimated savings: ~27 MB rlib, ~5-6 MB in final binary**

### Phase 2: Native PNG (drop biggest image crate feature)
- Implement native PNG decode (DEFLATE + filtering + interlacing)
- Implement native PNG encode (DEFLATE compression + filtering)
- Drop `image` crate `png` feature
- **Estimated savings: ~1 MB + progress toward dropping image crate**

### Phase 3: Native GIF + TIFF
- Native GIF decode/encode (LZW compression)
- Native TIFF decode/encode (multiple compression modes)
- Drop `image` crate `gif` and `tiff` features
- **Estimated savings: ~2.5 MB**

### Phase 4: Native remaining formats
- ICO, TGA, HDR, DDS decode — simple formats
- Drop `image` crate entirely
- **Estimated savings: ~7.5 MB (image crate fully removed)**

### Phase 5: Replace standalone external codecs
- Native JXL decode (replace jxl-oxide) — complex
- Native JP2 decode (replace justjp2) — complex
- Native AVIF via own AV1 decode (replace rav1e/ravif) — very complex
- **Low priority — these are already feature-gated after Phase 1**

## Target Size After Full Native Coverage

| Phase | Raw WASM | Gzipped |
|-------|----------|---------|
| Current | 11.3 MB | 3.4 MB |
| After Phase 1 (feature-gate) | ~6-7 MB | ~2.0 MB |
| After Phase 2 (native PNG) | ~5-6 MB | ~1.8 MB |
| After Phase 4 (drop image crate) | ~4-5 MB | ~1.5 MB |
| After Phase 5 (all native) | ~3-4 MB | ~1.0 MB |
