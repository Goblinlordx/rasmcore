# Feature Parity Matrix — rasmcore vs libvips vs ImageMagick

> Living document. Updated: 2026-03-28
> Each row is a potential implementation track.

## Legend

| Symbol | Meaning |
|--------|---------|
| done | Implemented and tested in rasmcore |
| partial | Partially implemented (limited params or formats) |
| planned | Tracked for implementation |
| - | Not planned / out of scope |

**Priority tiers:**
- **P0** — Critical for adoption (blocks real-world use)
- **P1** — High value (most users expect this)
- **P2** — Medium (power users, advanced workflows)
- **P3** — Low (niche, artistic effects)

---

## 1. Image Formats — Decode (Read)

| Format | rasmcore | libvips | ImageMagick | Priority | Notes |
|--------|----------|---------|-------------|----------|-------|
| PNG | done | R/W | R/W | - | Via `image` crate |
| JPEG | done | R/W | R/W | - | Via `image` crate |
| GIF | done | R/W | R/W | - | Decode only, no animation |
| WebP | done | R/W | R/W | - | Via `image` crate |
| BMP | done | via magick | R/W | - | Via `image` crate |
| TIFF | done | R/W | R/W | - | Via `image` crate |
| AVIF | done | R/W (via HEIF) | R/W | - | Via `image` crate |
| QOI | done | - | - | - | Unique to rasmcore |
| ICO | done | via magick | R/W | - | Via `image` crate |
| HEIF/HEIC | - | R/W | R/W | P1 | Needed for Apple ecosystem |
| JPEG XL (JXL) | - | R/W | R/W | P1 | Modern replacement for JPEG |
| SVG | - | R (librsvg) | R/W | P2 | Rasterization only |
| PDF | - | R (poppler) | R/W | P2 | Page rasterization |
| PSD | - | via magick | R/W | P3 | Photoshop files |
| EXR | - | R | R/W | P3 | HDR/VFX |
| Raw (CR2, NEF, ARW) | - | R (libraw) | R | P3 | Camera raw |
| PPM/PGM/PBM | - | R/W | R/W | P3 | Simple interchange |
| JPEG 2000 | - | R/W | R/W | P3 | Medical/satellite imaging |
| TGA | - | via magick | R/W | P3 | Legacy game art |
| HDR (Radiance) | - | R/W | R/W | P3 | HDR imaging |

**Summary:** 9/20+ decode formats. Missing P1: HEIF, JXL.

---

## 2. Image Formats — Encode (Write)

| Format | rasmcore | libvips | ImageMagick | Priority | Notes |
|--------|----------|---------|-------------|----------|-------|
| PNG | done | R/W | R/W | - | Per-format config in progress |
| JPEG | done | R/W | R/W | - | Per-format config in progress |
| WebP | done | R/W | R/W | - | Per-format config in progress |
| GIF | - | R/W | R/W | P0 | Very common output format |
| TIFF | - | R/W | R/W | P1 | Publishing, scanning, GIS |
| AVIF | - | R/W | R/W | P1 | Modern web format, great compression |
| HEIF/HEIC | - | R/W | R/W | P1 | Apple ecosystem |
| JPEG XL (JXL) | - | R/W | R/W | P1 | Modern replacement, progressive |
| BMP | - | via magick | R/W | P3 | Legacy |
| ICO | - | via magick | R/W | P3 | Favicon generation |
| SVG | - | - | R/W | P3 | Vector output |
| PDF | - | via magick | R/W | P3 | Document output |

**Summary:** 3/12+ encode formats. Missing P0: GIF. Missing P1: TIFF, AVIF, HEIF, JXL.

---

## 3. Resize / Resampling

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Nearest neighbor | done | yes | yes | - | |
| Bilinear | done | yes (linear) | yes (triangle) | - | |
| Bicubic (Catmull-Rom) | done | yes (cubic) | yes (catrom) | - | |
| Lanczos3 | done | yes | yes | - | |
| Lanczos2 | - | yes | yes | P2 | |
| Mitchell-Netravali | - | yes (mitchell) | yes (mitchell) | P2 | |
| MKS 2013/2021 | - | yes | - | P3 | Magic Kernel Sharp |
| Nohalo | - | yes | - | P3 | Edge-preserving |
| Shrink-on-load | - | yes (JPEG/TIFF/JP2K) | yes (JPEG) | P1 | Major perf optimization |
| Thumbnail (smart) | - | yes | yes | P1 | Resize+crop in one step |

**Summary:** 4/10 resize algorithms. Missing P1: shrink-on-load, smart thumbnail.

---

## 4. Geometric Transforms

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Resize | done | yes | yes | - | |
| Crop (extract area) | done | yes | yes | - | |
| Rotate 90/180/270 | done | yes (rot) | yes | - | |
| Flip H/V | done | yes | yes (flip/flop) | - | |
| Pixel format convert | done | yes | yes | - | |
| Arbitrary rotation | - | yes (rotate) | yes (rotate) | P1 | Any angle, with background fill |
| Affine transform | - | yes (affine) | yes (affine) | P2 | General 2D transform matrix |
| Smart crop | - | yes (smartcrop) | yes (trim) | P1 | Content-aware crop |
| Embed/extend (pad) | - | yes (embed) | yes (extent) | P1 | Add borders/padding |
| Auto-orient (EXIF) | - | yes (autorot) | yes (auto-orient) | P0 | Rotate based on EXIF |
| Shear | - | yes (via affine) | yes (shear) | P3 | |
| Perspective/distort | - | - | yes (distort) | P3 | |
| Liquid rescale | - | - | yes (liquid-rescale) | P3 | Seam carving |
| Trim/autocrop | - | yes (find_trim) | yes (trim) | P2 | Remove uniform borders |

**Summary:** 5/14 geometric ops. Missing P0: auto-orient. Missing P1: arbitrary rotation, smart crop, embed/pad.

---

## 5. Filters & Effects

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Gaussian blur | done | yes (gaussblur) | yes (gaussian) | - | |
| Sharpen (unsharp mask) | done | yes (sharpen) | yes (unsharp) | - | |
| Brightness adjust | done | yes (linear) | yes (brightness) | - | |
| Contrast adjust | done | yes (linear) | yes (contrast) | - | |
| Grayscale | done | yes (colourspace) | yes (colorspace) | - | |
| Gamma correction | - | yes (gamma) | yes (gamma) | P1 | Essential color tool |
| Levels/curves | - | yes (maplut) | yes (level) | P1 | Tone mapping |
| Invert/negate | - | yes (invert) | yes (negate) | P1 | Simple but essential |
| Histogram equalize | - | yes (hist_equal) | yes (equalize) | P1 | Auto-contrast |
| CLAHE (local contrast) | - | yes (hist_local) | yes (CLAHE) | P2 | Adaptive local contrast |
| Canny edge detect | - | yes (canny) | yes (canny) | P2 | |
| Sobel edge detect | - | yes (sobel) | yes (edge) | P2 | |
| Median filter | - | yes (median/rank) | yes (median) | P2 | Noise reduction |
| Custom convolution | - | yes (conv) | yes (convolve) | P2 | User-defined kernels |
| Morphology (erode/dilate) | - | yes (morph) | yes (morphology) | P2 | |
| Threshold (binary) | - | yes (threshold) | yes (threshold) | P1 | Binary image creation |
| Posterize | - | - | yes (posterize) | P3 | Reduce color levels |
| Solarize | - | - | yes (solarize) | P3 | |
| Oil paint | - | - | yes (oil-paint) | P3 | Artistic |
| Charcoal | - | - | yes (charcoal) | P3 | Artistic |
| Emboss | - | yes (via conv) | yes (emboss) | P3 | |
| Vignette | - | - | yes (vignette) | P3 | |
| Motion blur | - | - | yes (motion-blur) | P3 | |
| Noise generation | - | yes (gaussnoise) | yes (noise) | P3 | |

**Summary:** 5/24 filters. Missing P1: gamma, levels, invert, histogram equalize, threshold.

---

## 6. Color Management

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| sRGB | done | yes | yes | - | Only color space supported |
| ICC profile read | - | yes (icc_import) | yes | P0 | Correct color reproduction |
| ICC profile embed | - | yes (icc_export) | yes | P0 | Output color accuracy |
| ICC transform | - | yes (icc_transform) | yes | P1 | Device-to-device |
| CMYK | - | yes | yes | P1 | Print workflows |
| Lab (CIELAB) | - | yes | yes | P2 | Perceptual operations |
| HSV/HSL | - | yes | yes | P2 | Intuitive color manipulation |
| XYZ | - | yes | yes | P3 | Color science |
| Linear sRGB (scRGB) | partial (type exists) | yes | yes | P1 | HDR, compositing |
| Display P3 | partial (type exists) | yes | - | P2 | Wide gamut displays |
| BT.709 / BT.2020 | partial (type exists) | yes | yes | P2 | Video color spaces |
| Oklab/Oklch | - | yes | - | P2 | Modern perceptual space |
| Color difference (dE) | - | yes (dE76/dE00) | - | P3 | Quality metrics |

**Summary:** 1/13 color ops. Missing P0: ICC read/embed. Missing P1: ICC transform, CMYK, linear sRGB.

---

## 7. Compositing & Blending

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Alpha composite (over) | - | yes | yes | P0 | Basic layer compositing |
| Porter-Duff modes | - | yes (13 modes) | yes | P1 | Full compositing algebra |
| Blend modes (multiply, screen, overlay...) | - | yes (11 modes) | yes (40+ modes) | P1 | Photo editing |
| Watermark overlay | - | yes (composite) | yes (composite) | P1 | Common use case |
| Image concatenation | - | yes (arrayjoin) | yes (append) | P2 | Side-by-side, grid |
| Montage/contact sheet | - | - | yes (montage) | P3 | |

**Summary:** 0/6 compositing ops. Missing P0: alpha composite.

---

## 8. Drawing & Text

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Draw rectangle | - | yes (draw_rect) | yes | P2 | |
| Draw line | - | yes (draw_line) | yes | P2 | |
| Draw circle | - | yes (draw_circle) | yes | P2 | |
| Text rendering | - | yes (text) | yes (annotate) | P2 | |
| Flood fill | - | yes (draw_flood) | yes (floodfill) | P3 | |
| Bezier/path | - | - | yes (draw) | P3 | |

**Summary:** 0/6 drawing ops. All P2-P3.

---

## 9. Metadata & I/O

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| EXIF read | - | yes | yes | P0 | Basic image metadata |
| EXIF write/strip | - | yes | yes | P1 | Privacy, file size |
| XMP read/write | - | - | yes | P3 | Extended metadata |
| IPTC read/write | - | - | yes | P3 | News/stock photo |
| Multi-page/frame read | - | yes | yes | P1 | TIFF pages, GIF frames |
| Animation support | - | yes (GIF, WebP) | yes (GIF, WebP, APNG) | P2 | Animated images |

**Summary:** 0/6 metadata ops. Missing P0: EXIF read.

---

## 10. Architecture & Performance

| Capability | rasmcore | libvips | ImageMagick | Priority | Notes |
|------------|----------|---------|-------------|----------|-------|
| Streaming/scanline processing | - | yes (demand-driven) | - | P1 | Avoids full-image-in-memory |
| SIMD acceleration | - | yes (Highway) | yes (OpenMP) | P2 | 3-4x speedup on filters |
| Shrink-on-load (JPEG DCT) | - | yes | yes | P1 | Major perf for thumbnailing |
| Thread pool | - | yes | yes (OpenMP) | P2 | Parallel tile processing |
| Operation caching | - | yes | yes (pixel cache) | P3 | Avoid recomputation |
| Chainable pipeline API | - | yes (method chaining) | yes (mogrify) | P1 | Ergonomic multi-op workflows |
| WASM Component Model | done | - | - | - | Unique advantage |
| Multi-language SDKs | done | - | - | - | Via WIT bindings |
| Small bundle size | done | - (15MB) | - (15MB) | - | Unique advantage |
| No SharedArrayBuffer req | done | - (required) | done | - | Browser compatibility |

---

## Priority Summary

### P0 — Critical for Adoption
1. **GIF encode** — Very common output format
2. **Auto-orient (EXIF)** — Images display rotated without this
3. **ICC profile read/embed** — Color accuracy for any serious use
4. **Alpha composite (over)** — Basic layer compositing
5. **EXIF metadata read** — Basic image info (dimensions from header, orientation, camera)

### P1 — High Value
6. AVIF encode — Modern web format, superior compression
7. TIFF encode — Publishing, scanning, GIS workflows
8. HEIF/HEIC decode+encode — Apple ecosystem
9. JXL decode+encode — Next-gen format
10. Gamma correction — Essential color tool
11. Invert/negate — Simple but essential filter
12. Threshold (binary) — Binary image creation
13. Levels/curves — Tone mapping
14. Histogram equalize — Auto-contrast
15. Arbitrary angle rotation — Common transform
16. Smart crop — Content-aware crop
17. Embed/pad — Add borders
18. Shrink-on-load — Major perf for thumbnailing
19. Streaming pipeline — Architectural improvement for large images
20. Chainable pipeline API — Ergonomic SDK improvement
21. Porter-Duff blend modes — Photo editing compositing
22. Watermark overlay — Common use case
23. EXIF write/strip — Privacy, file size
24. Multi-page read — TIFF pages, GIF frames
25. CMYK color space — Print workflows
26. Linear sRGB — HDR, compositing

### P2 — Medium
27. Lanczos2, Mitchell filters
28. CLAHE, Canny, Sobel, median, convolution, morphology
29. HSV/HSL, Lab, Oklab color spaces
30. Image concatenation, drawing ops, text
31. Affine transform, trim/autocrop
32. Animation support, SIMD, thread pool

### P3 — Low / Niche
33. Artistic effects (oil paint, charcoal, emboss, solarize)
34. EXR, Raw, PSD, JPEG 2000, HDR formats
35. Flood fill, bezier paths
36. XMP/IPTC metadata
37. Perspective distort, liquid rescale

---

## Recommended Implementation Order

**Phase 1 — Close P0 gaps (5 items)**
These block real-world adoption. Ship these before marketing.

**Phase 2 — Core P1 formats (items 6-9)**
AVIF, TIFF, HEIF, JXL encode — each is a separate per-format codec track.

**Phase 3 — P1 operations (items 10-16)**
Essential filters and transforms that users expect from any image library.

**Phase 4 — P1 architecture (items 17-20)**
Streaming, shrink-on-load, pipeline API — these are cross-cutting improvements.

**Phase 5 — P1 compositing & color (items 21-26)**
Blending, ICC profiles, CMYK — unlocks photo editing and print workflows.

**Phase 6+ — P2/P3 as demand warrants**
Prioritize based on user feedback and competitive pressure.
