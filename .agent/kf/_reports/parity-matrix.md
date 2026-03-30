# Feature Parity Matrix — rasmcore vs libvips vs ImageMagick

> Living document. Updated: 2026-03-30
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
| JPEG | done | R/W | R/W | - | Native (rasmcore-jpeg) + `image` crate |
| GIF | done | R/W | R/W | - | Static only, no animation |
| WebP | done | R/W | R/W | - | Via `image-webp` crate |
| BMP | done | via magick | R/W | - | Native (rasmcore-bmp) |
| TIFF | done | R/W | R/W | - | Via `tiff` crate |
| AVIF | done | R/W (via HEIF) | R/W | - | Via `image` crate |
| QOI | done | - | - | - | Native (rasmcore-qoi). Unique to rasmcore |
| ICO | done | via magick | R/W | - | Via `image` crate |
| TGA | done | via magick | R/W | - | Native (rasmcore-tga) |
| HDR/Radiance | done | R/W | R/W | - | Via `image` crate |
| PNM (PPM/PGM/PBM/PAM) | done | R/W | R/W | - | Native (rasmcore-pnm) |
| OpenEXR | done | R | R/W | - | Via `exr` crate |
| DDS | done | via magick | R/W | - | Native encoder/decoder |
| JPEG XL (JXL) | done | R/W | - | - | Via `jxl-oxide` (pure Rust) |
| JPEG 2000 (JP2) | done | R/W | R/W | - | Via `justjp2` (pure Rust) |
| HEIF/HEIC | done | R/W | R/W | - | Feature-gated (nonfree-hevc) |
| FITS | done | R/W | R/W | - | Native (rasmcore-fits) |
| SVG | done | R (librsvg) | R/W | - | Via `resvg` (pure Rust rasterizer) |
| Animated GIF | partial | R/W | R/W | P1 | First frame only; no multi-frame |
| Animated WebP | partial | R/W | R/W | P1 | First frame only; no multi-frame |
| PSD | - | via magick | R/W | P2 | Photoshop files |
| PDF | - | R (poppler) | R/W | P2 | Document format, different domain |
| Raw (CR2, NEF, ARW) | - | R (libraw) | R | P3 | No pure Rust decoder |

**Summary:** 18/24 decode formats fully implemented. Only animation and niche formats remain.

---

## 2. Image Formats — Encode (Write)

| Format | rasmcore | libvips | ImageMagick | Priority | Notes |
|--------|----------|---------|-------------|----------|-------|
| PNG | done | R/W | R/W | - | Via `image` crate |
| JPEG | done | R/W | R/W | - | Native (rasmcore-jpeg) |
| WebP | done | R/W | R/W | - | Native (rasmcore-webp, lossy+lossless) |
| GIF | done | R/W | R/W | - | Via `image` crate + NeuQuant |
| TIFF | done | R/W | R/W | - | Via `tiff` crate (LZW/Deflate/PackBits) |
| AVIF | done | R/W | R/W | - | Via `image` crate (rav1e backend) |
| BMP | done | via magick | R/W | - | Native (rasmcore-bmp) |
| ICO | done | via magick | R/W | - | Via `image` crate |
| QOI | done | - | - | - | Native (rasmcore-qoi). Unique to rasmcore |
| TGA | done | via magick | R/W | - | Native (rasmcore-tga) |
| HDR/Radiance | done | R/W | R/W | - | Native encoder |
| PNM | done | R/W | R/W | - | Native (rasmcore-pnm) |
| OpenEXR | done | - | R/W | - | Via `exr` crate |
| DDS | done | via magick | R/W | - | Native encoder |
| JPEG 2000 (JP2) | done | R/W | R/W | - | Via `justjp2` |
| FITS | done | R/W | R/W | - | Native (rasmcore-fits) |
| HEIF/HEIC | done | R/W | R/W | - | Feature-gated (nonfree-hevc) |
| JPEG XL (JXL) | partial | R/W | - | P1 | Decode done; encode scaffold only |

**Summary:** 16/18 encode formats fully implemented. Only JXL encode scaffold remains.

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
| Thumbnail (smart) | done | yes | yes | - | smart_crop in content_aware.rs |

**Summary:** 5/10 resize algorithms. Missing P1: shrink-on-load.

---

## 4. Geometric Transforms

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Resize | done | yes | yes | - | fast_image_resize (SIMD) |
| Crop (extract area) | done | yes | yes | - | |
| Rotate 90/180/270 | done | yes (rot) | yes | - | 8x8 block transpose |
| Flip H/V | done | yes | yes (flip/flop) | - | |
| Pixel format convert | done | yes | yes | - | |
| Auto-orient (EXIF) | done | yes (autorot) | yes (auto-orient) | - | auto_orient + auto_orient_from_exif |
| Arbitrary rotation | done | yes (rotate) | yes (rotate) | - | rotate_arbitrary with bilinear interp |
| Embed/extend (pad) | done | yes (embed) | yes (extent) | - | pad(top, left, bottom, right, fill) |
| Trim/autocrop | done | yes (find_trim) | yes (trim) | - | trim with threshold |
| Affine transform | done | yes (affine) | yes (affine) | - | 2x3 matrix |
| Lens undistort | done | - | - | - | OpenCV-compatible distortion correction |
| Smart crop | done | yes (smartcrop) | yes (trim) | - | Content-aware crop (smart_crop.rs) |
| Seam carve (width) | done | - | yes (liquid-rescale) | - | Content-aware resize |
| Seam carve (height) | done | - | yes (liquid-rescale) | - | Content-aware resize |
| Shear | - | yes (via affine) | yes (shear) | P3 | Can be done via affine transform |
| Perspective/distort | done | - | yes (distort) | - | perspective_warp (3x3 homography) |

**Summary:** 14/16 geometric ops. Only shear missing (P3, achievable via existing affine).

---

## 5. Filters & Effects

### 5a. Registered Filters (56+ total — visible to WASM/SDK/UI)

| Operation | rasmcore | libvips | ImageMagick | Category | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Gaussian blur | done | yes (gaussblur) | yes (gaussian) | spatial | SIMD via libblur |
| Bokeh blur | done | - | - | spatial | Disc/hex/octagon/star shapes |
| Motion blur | done | - | yes (motion-blur) | spatial | Directional blur |
| Zoom blur | done | - | - | spatial | Radial zoom from center |
| Gaussian blur (OpenCV) | done | - | - | spatial | Separate sigma_x/sigma_y |
| Sharpen (unsharp mask) | done | yes (sharpen) | yes (unsharp) | spatial | |
| Convolve (custom kernel) | done | yes (conv) | yes (convolve) | spatial | |
| Median filter | done | yes (median/rank) | yes (median) | spatial | Histogram-based |
| Bilateral filter | done | yes (bilateral) | - | spatial | OpenCV parity (exact) |
| Guided filter | done | - | - | spatial | He et al. 2010 |
| Displacement map | done | - | yes (distort) | spatial | |
| Brightness | done | yes (linear) | yes (brightness) | adjustment | LUT-collapsible |
| Contrast | done | yes (linear) | yes (contrast) | adjustment | LUT-collapsible |
| Hue rotate | done | yes (colourspace) | yes | color | HSV-based |
| Saturate | done | yes (colourspace) | yes (modulate) | color | |
| Sepia | done | - | yes (sepia-tone) | color | |
| Colorize | done | - | yes (colorize) | color | |
| Sobel edge | done | yes (sobel) | yes (edge) | edge | |
| Scharr edge | done | - | - | edge | More accurate than Sobel |
| Laplacian edge | done | - | yes (laplacian) | edge | |
| Canny edge | done | yes (canny) | yes (canny) | edge | |
| CLAHE | done | yes (hist_local) | yes (CLAHE) | enhancement | OpenCV parity (MAE < 0.5) |
| Dehaze | done | - | - | enhancement | Dark channel prior |
| Clarity | done | - | - | enhancement | Frequency separation |
| Frequency low/high | done | - | - | enhancement | Gaussian decomposition |
| Pyramid detail remap | done | - | - | enhancement | Laplacian pyramid |
| Vignette (gaussian) | done | - | yes (vignette) | enhancement | |
| Vignette (power-law) | done | - | - | enhancement | Alternative falloff |
| Retinex SSR/MSR/MSRCR | done | - | - | enhancement | 3 variants |
| NLM denoise | done | - | yes (non-local-means) | enhancement | |
| Premultiply/unpremultiply | done | yes (premultiply) | - | alpha | |
| Morphology (7 ops) | done | yes (morph) | yes (morphology) | morphology | Erode/dilate/open/close/gradient/tophat/blackhat |
| Threshold binary | done | yes (threshold) | yes (threshold) | threshold | |
| Adaptive threshold | done | - | yes (-adaptive-threshold) | threshold | Mean or Gaussian |
| Perlin noise | done | yes (gaussnoise) | yes (+noise) | generator | |
| Simplex noise | done | - | - | generator | |
| Flood fill | done | yes (draw_flood) | yes (floodfill) | tool | |
| Perspective warp | done | - | yes (distort) | advanced | 3x3 homography |
| Perspective correct | done | - | - | advanced | |
| Pixelate | done | - | yes (scale) | effect | IM parity: EXACT (MAE 0.00) |
| Halftone | done | - | - | effect | CMYK dot screen, no IM equivalent |
| Swirl | done | - | yes (swirl) | distortion | IM parity: EXACT (MAE 0.001) |
| Spherize | done | - | - | distortion | Bulge/pinch, no IM equivalent |
| Barrel distortion | done | - | yes (distort Barrel) | distortion | IM parity: CLOSE (MAE 0.28) |
| Kuwahara | done | - | yes (kuwahara) | spatial | IM parity: interior bit-exact (MAE 0.33) |
| Rank filter | done | yes (rank) | yes (statistic) | spatial | IM parity: EXACT (MAE 0.00) |

### 5b. Implemented But Unregistered (invisible to WASM/SDK/UI)

These functions exist and are tested but lack `#[register_filter]` annotations. Tracked for registration in `register-filters-core` and `register-filters-pro` tracks.

| Operation | Source File | Category | Notes |
|-----------|-----------|----------|-------|
| **gamma** | point_ops.rs | adjustment | LUT-collapsible |
| **invert/negate** | point_ops.rs | adjustment | LUT-collapsible |
| **posterize** | point_ops.rs | adjustment | LUT-collapsible |
| **equalize** | histogram.rs | enhancement | Histogram equalization |
| **normalize** | histogram.rs | enhancement | Contrast normalization |
| **auto_level** | histogram.rs | enhancement | Auto contrast stretch |
| **histogram_match** | histogram.rs | enhancement | Match target histogram |
| **contrast_stretch** | histogram.rs | enhancement | |
| **otsu_threshold** | filters.rs | threshold | Auto-compute + apply |
| **triangle_threshold** | filters.rs | threshold | Auto-compute + apply |
| **grayscale** | filters.rs | color | RGB to Gray8 |
| **flatten** | filters.rs | alpha | Composite over solid bg |
| **add_alpha / remove_alpha** | filters.rs | alpha | Channel manipulation |
| **blend (16 modes)** | filters.rs | compositing | Multiply, Screen, Overlay, Darken, Lighten, SoftLight, HardLight, Difference, Exclusion, ColorDodge, ColorBurn, VividLight, LinearDodge, LinearBurn, LinearLight, PinLight |
| **ASC CDL** | color_grading.rs | grading | Industry-standard color correction |
| **lift/gamma/gain** | color_grading.rs | grading | Professional 3-way grading |
| **split toning** | color_grading.rs | grading | Highlight/shadow tinting |
| **curves** | color_grading.rs | grading | Spline-based tone curves |
| **tonemap Reinhard** | color_grading.rs | tonemapping | HDR to SDR |
| **tonemap Drago** | color_grading.rs | tonemapping | HDR to SDR |
| **tonemap Filmic** | color_grading.rs | tonemapping | HDR to SDR |
| **film grain** | color_grading.rs | effect | Procedural grain overlay |
| **white balance (gray world)** | color_spaces.rs | color | Auto white balance |
| **white balance (temperature)** | color_spaces.rs | color | Manual Kelvin/tint |
| **quantize (median cut)** | quantize.rs | color | Color reduction |
| **dither (Floyd-Steinberg)** | quantize.rs | color | Error diffusion |
| **dither (ordered/Bayer)** | quantize.rs | color | Ordered dithering |
| **smart_crop** | smart_crop.rs | transform | Content-aware crop |
| **seam_carve (width/height)** | content_aware.rs | transform | Content-aware resize |
| **selective_color** | content_aware.rs | color | Photoshop-style selective adjustment |
| **inpaint** | inpainting.rs | tool | Multi-image (mask required) |
| **alpha_composite_over** | composite.rs | compositing | Porter-Duff over |
| **mertens_fusion** | filters.rs | HDR | Multi-image exposure fusion |
| **debevec_hdr_merge** | filters.rs | HDR | Multi-image HDR recovery |

**Total implemented operations: 49 registered + ~34 unregistered = 83+ filters/operations.**

### 5c. Truly Missing Filters

No major filter gaps remain. All standard ImageMagick spatial, effect, and
distortion filters now have rasmcore equivalents with reference parity tests.

**Summary:** 0 truly missing filters. 56+ registered filters covering all categories.

---

## 6. Color Management

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| sRGB | done | yes | yes | - | Default color space |
| ICC profile extract (JPEG) | done | yes | yes | - | color.rs |
| ICC profile extract (PNG) | done | yes | yes | - | color.rs |
| ICC to sRGB transform | done | yes (icc_transform) | yes | - | icc_to_srgb() |
| Lab (CIELAB) | done | yes | yes | - | rgb_to_lab / lab_to_rgb + image-level |
| Oklab/Oklch | done | yes | - | - | Direct + via XYZ paths |
| ProPhoto RGB | done | yes | yes | - | Bidirectional conversion |
| Adobe RGB | done | yes | yes | - | Bidirectional conversion |
| LCH | done | yes | yes | - | lab_to_lch / lch_to_lab |
| Luv | done | yes | - | - | via XYZ |
| XYZ | done | yes | yes | - | Via Lab/Oklab conversions |
| Bradford chromatic adapt | done | yes | yes | - | Illuminant adaptation |
| Delta E (76, 94, 2000) | done | yes (dE76/dE00) | - | - | All 3 standard metrics |
| White balance (gray world) | done | - | - | - | Auto white balance |
| White balance (temperature) | done | - | - | - | Kelvin + tint control |
| 3D Color LUT | done | yes | yes | - | Compose, absorb 1D pre/post |
| HSV/HSL | done | yes | yes | - | Via hue_rotate, saturate operations |
| Linear sRGB | partial | yes | yes | P2 | Type exists, limited use |
| Display P3 | partial | yes | - | P2 | Type exists, limited use |
| BT.709 / BT.2020 | partial | yes | yes | P2 | Type exists, HEVC uses BT.709 |
| CMYK (full pipeline) | - | yes | yes | P1 | Print workflows |

**Summary:** 17/21 color operations. Missing P1: full CMYK pipeline.

---

## 7. Compositing & Blending

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Alpha composite (over) | done | yes | yes | - | Porter-Duff over (composite.rs) |
| Blend modes (16 modes) | done | yes (11 modes) | yes (40+ modes) | - | Multiply, Screen, Overlay, Darken, Lighten, SoftLight, HardLight, Difference, Exclusion, ColorDodge, ColorBurn, VividLight, LinearDodge, LinearBurn, LinearLight, PinLight |
| Flatten (alpha over bg) | done | yes | yes | - | flatten() with bg_r/g/b |
| Premultiply/unpremultiply | done | yes | - | - | Registered filters |
| Image concatenation | - | yes (arrayjoin) | yes (append) | P2 | Side-by-side, grid |
| Montage/contact sheet | - | - | yes (montage) | P3 | |

**Summary:** 4/6 compositing ops. rasmcore has 16 blend modes vs libvips 11.

---

## 8. Drawing & Text

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Draw rectangle | - | yes (draw_rect) | yes | P2 | |
| Draw line | - | yes (draw_line) | yes | P2 | |
| Draw circle | - | yes (draw_circle) | yes | P2 | |
| Text rendering | - | yes (text) | yes (annotate) | P2 | |
| Flood fill | done | yes (draw_flood) | yes (floodfill) | - | Registered filter |

**Summary:** 1/5 drawing ops. All missing are P2.

---

## 9. Metadata & I/O

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| EXIF read | done | yes | yes | - | read_exif() + has_exif() |
| EXIF write | done | yes | yes | - | write_exif() |
| XMP read/write | done | - | yes | - | parse_xmp() + serialize_xmp() |
| IPTC read/write | done | - | yes | - | parse_iptc() + serialize_iptc() |
| MetadataSet (unified API) | done | - | - | - | metadata_read(), metadata_dump_json() |
| Multi-page/frame read | - | yes | yes | P1 | TIFF pages, GIF/WebP frames |
| Animation support | - | yes (GIF, WebP) | yes (GIF, WebP, APNG) | P1 | Research track created |

**Summary:** 5/7 metadata ops. Missing P1: multi-page/animation.

---

## 10. Architecture & Performance

| Capability | rasmcore | libvips | ImageMagick | Priority | Notes |
|------------|----------|---------|-------------|----------|-------|
| Demand-driven tile pipeline | done | yes (demand-driven) | - | - | Pull-based with spatial cache |
| SIMD acceleration | done | yes (Highway) | yes (OpenMP) | - | libblur, fast_image_resize, explicit WASM128 |
| WASM Component Model | done | - | - | - | Unique advantage |
| Multi-language SDKs | done | - | - | - | Via WIT bindings |
| Small bundle size | done | - (15MB) | - (15MB) | - | Unique advantage |
| LUT fusion optimizer | done | - | - | - | Collapses consecutive per-pixel ops |
| No-op filter elision | done | - | - | - | Skip identity-parameter filters |
| Shrink-on-load (JPEG DCT) | - | yes | yes | P1 | Major perf for thumbnailing |
| Thread pool | - | yes | yes (OpenMP) | P2 | Parallel tile processing |

---

## 11. ImageMagick Format Catalog Analysis

ImageMagick claims ~260 format entries. The actual breakdown:

| Category | Count | Examples | Relevance |
|----------|-------|---------|-----------|
| **Core raster** | ~12 | PNG, JPEG, WebP, GIF, TIFF, BMP, TGA, ICO | Must-have (all done in rasmcore) |
| **Modern web** | ~6 | AVIF, HEIC, JXL, WebP, JP2, UltraHDR | Critical (5/6 done) |
| **Professional** | ~12 | EXR, DPX, CIN, PSD, DDS, SGI, HDR | Niche but valued (4/12 done) |
| **Camera RAW** | ~30 | CR2, NEF, ARW, DNG (all via libraw delegate) | All read-only, all C delegates |
| **Scientific** | ~6 | FITS, DICOM, Analyze, VICAR | Niche (FITS done) |
| **Legacy/historical** | ~30 | PCX, XPM, WBMP, Sun Raster, MacPaint, ZX Spectrum, PalmDB | Obsolete — skip |
| **Vector/document** | ~10 | SVG, PDF, PS, EPS, AI | Different domain (SVG rasterize done) |
| **Video containers** | ~10 | AVI, MP4, MOV (frame extraction via ffmpeg) | Not image formats |
| **IM-internal/aliases** | ~60 | MIFF, MPC, PNG8/24/32/48/64, raw pixel dumps (RGB/RGBA/BGRA/CMYK/GRAY) | Not real formats |
| **Pseudo-formats** | ~15 | xc:, gradient:, plasma:, label:, pattern: | Generators, not files |
| **Output-only** | ~12 | HTML, JSON, YAML, Braille, fax, SIXEL | Serialization |
| **Fonts** | ~5 | TTF, OTF (rasterize glyphs) | Not image formats |

**Real distinct image formats: ~80-90.** Of those, ~30 are legacy/historical with near-zero demand.

libvips supports ~25 format families natively (JPEG, PNG, WebP, GIF, TIFF, HEIC, AVIF, JXL, JP2, PPM, HDR, SVG, PDF, FITS, EXR, MATLAB, Analyze, Camera RAW via libraw, DeepZoom, OpenSlide, UltraHDR, CSV, VIPS native), falling back to IM for the rest.

**rasmcore coverage: 20/26 meaningful Tier 1-3 format families.**

---

## Priority Summary (Updated 2026-03-30)

### P0 — Critical for Adoption
All P0 items are now DONE:
- ~~GIF encode~~ done
- ~~Auto-orient (EXIF)~~ done
- ~~ICC profile read/embed~~ done
- ~~Alpha composite (over)~~ done
- ~~EXIF metadata read~~ done

### P1 — High Value
1. **Shrink-on-load** — Major perf optimization for thumbnailing
2. **Multi-page/frame read** — TIFF pages, GIF/WebP frames
3. **Animation support** — Animated GIF/WebP/APNG (research track created)
4. **CMYK full pipeline** — Print workflows
5. **JXL encode** — Next-gen format (scaffold exists)
6. Register ~30 unregistered filters (2 tracks created)

### P2 — Medium
7. Lanczos2, Mitchell-Netravali resize filters
8. Draw primitives (line, rect, circle, text)
9. Image concatenation / arrayjoin
10. Linear sRGB / Display P3 full pipeline
11. Thread pool for parallel tile processing

### P3 — Low / Niche
12. Artistic effects (solarize, emboss, oil paint, charcoal)
13. Shear transform
14. Camera RAW (blocked by no pure-Rust decoder)
15. PSD, DPX, CIN, DICOM

---

## rasmcore Unique Advantages

| Advantage | Detail |
|-----------|--------|
| **Pure Rust / WASM** | Compiles to wasm32-wasip2, runs in browsers and edge compute |
| **No C dependencies** | Core codecs are pure Rust — no FFI vulnerabilities |
| **QOI format** | Native support — neither IM nor vips has it |
| **DDS read+write** | Both IM and vips have limited DDS support |
| **16 blend modes** | More than libvips (11), validated against vips + ImageMagick |
| **Professional grading** | ASC CDL, lift/gamma/gain, curves — uncommon in image libraries |
| **Content-aware ops** | Smart crop + seam carve + selective color + inpainting |
| **Color science** | Lab, Oklab, ProPhoto, Adobe RGB, LCH, Luv, Delta E (76/94/2000) |
| **3D Color LUTs** | Compose, absorb 1D, bake from grading ops |
| **Component Model** | WIT interfaces enable cross-language composability |
| **83+ total operations** | 49 registered + 34 unregistered implementations |
