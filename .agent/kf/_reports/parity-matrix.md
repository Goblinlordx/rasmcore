# Feature Parity Matrix — rasmcore vs libvips vs ImageMagick

> Living document. Updated: 2026-03-31
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

### 5a. Registered Filters (128 total — visible to WASM/SDK/UI)

All previously unregistered filters have been registered via config-struct migration.
128 filters + 9 additional registrations (generators, compositors, mappers, encoders, decoders) = **137 total registered operations**.

| Operation | rasmcore | libvips | ImageMagick | Category | Notes |
|-----------|----------|---------|-------------|----------|-------|
| **Spatial** | | | | | |
| Gaussian blur | done | yes (gaussblur) | yes (gaussian) | spatial | SIMD via libblur |
| Bokeh blur | done | - | - | spatial | Disc/hex/octagon/star shapes |
| Motion blur | done | - | yes (motion-blur) | spatial | Directional blur |
| Zoom blur | done | - | - | spatial | Radial zoom from center |
| Spin blur | done | - | yes (radial-blur) | spatial | Rotational blur |
| Gaussian blur (OpenCV) | done | - | - | spatial | Separate sigma_x/sigma_y |
| Sharpen (unsharp mask) | done | yes (sharpen) | yes (unsharp) | spatial | |
| Convolve (custom kernel) | done | yes (conv) | yes (convolve) | spatial | |
| Median filter | done | yes (median/rank) | yes (median) | spatial | Histogram-based |
| Bilateral filter | done | yes (bilateral) | - | spatial | OpenCV parity (exact) |
| Guided filter | done | - | - | spatial | He et al. 2010 |
| Kuwahara | done | - | yes (kuwahara) | spatial | IM parity: interior bit-exact (MAE 0.33) |
| Rank filter | done | yes (rank) | yes (statistic) | spatial | IM parity: EXACT (MAE 0.00) |
| Displacement map | done | - | yes (distort) | spatial | |
| Tilt shift | done | - | - | spatial | Miniature/depth-of-field simulation |
| Lens blur | done | - | - | spatial | Disc-kernel bokeh |
| **Adjustment** | | | | | |
| Brightness | done | yes (linear) | yes (brightness) | adjustment | LUT-collapsible |
| Contrast | done | yes (linear) | yes (contrast) | adjustment | LUT-collapsible |
| Gamma | done | yes (gamma) | yes (gamma) | adjustment | LUT-collapsible |
| Levels | done | yes (linear) | yes (level) | adjustment | 5-param (black/white in/out + gamma) |
| Sigmoidal contrast | done | - | yes (sigmoidal-contrast) | adjustment | |
| Dodge | done | - | yes (dodge) | enhancement | Lighten shadows |
| Burn | done | - | yes (burn) | enhancement | Darken highlights |
| Shadow/highlight | done | - | - | enhancement | Frequency-separated recovery |
| **Color** | | | | | |
| Hue rotate | done | yes (colourspace) | yes | color | HSV-based |
| Saturate | done | yes (colourspace) | yes (modulate) | color | |
| Vibrance | done | - | - | color | Selective saturation |
| Sepia | done | - | yes (sepia-tone) | color | |
| Colorize | done | - | yes (colorize) | color | |
| Photo filter | done | - | - | color | Photographic filter simulation |
| Channel mixer | done | - | - | color | Channel routing/swapping |
| Gradient map | done | - | - | color | Luminance → color gradient |
| Sparse color | done | - | yes (sparse-color) | color | |
| Modulate | done | yes (colourspace) | yes (modulate) | color | Combined HSL modulation |
| Grayscale | done | yes (colourspace) | yes (type Grayscale) | color | RGB to Gray8 |
| Selective color | done | - | - | color | Photoshop-style hue-range adjustment |
| White balance (gray world) | done | - | - | color | Auto white balance |
| White balance (temperature) | done | - | yes (white-balance) | color | Kelvin + tint |
| Invert | done | yes (invert) | yes (negate) | adjustment | LUT-collapsible |
| Posterize | done | - | yes (posterize) | adjustment | LUT-collapsible |
| Solarize | done | - | yes (solarize) | effect | |
| **Edge Detection** | | | | | |
| Sobel edge | done | yes (sobel) | yes (edge) | edge | |
| Scharr edge | done | - | - | edge | More accurate than Sobel |
| Laplacian edge | done | - | yes (laplacian) | edge | |
| Canny edge | done | yes (canny) | yes (canny) | edge | |
| Hough lines | done | yes (hough_line) | - | edge | Probabilistic variant |
| Connected components | done | yes (labelregions) | yes | edge | Region labeling |
| **Enhancement** | | | | | |
| CLAHE | done | yes (hist_local) | yes (CLAHE) | enhancement | OpenCV parity (MAE < 0.5) |
| Equalize | done | yes (hist_equal) | yes (equalize) | enhancement | Histogram equalization |
| Normalize | done | yes (hist_norm) | yes (normalize) | enhancement | Contrast normalization |
| Auto level | done | - | yes (auto-level) | enhancement | Auto contrast stretch |
| Dehaze | done | - | - | enhancement | Dark channel prior |
| Clarity | done | - | - | enhancement | Frequency separation |
| Frequency low/high | done | - | - | enhancement | Gaussian decomposition |
| Pyramid detail remap | done | - | - | enhancement | Laplacian pyramid |
| Vignette (gaussian) | done | - | yes (vignette) | enhancement | |
| Vignette (power-law) | done | - | - | enhancement | Alternative falloff |
| Retinex SSR/MSR/MSRCR | done | - | - | enhancement | 3 variants |
| NLM denoise | done | - | yes (non-local-means) | enhancement | |
| **Threshold** | | | | | |
| Threshold binary | done | yes (threshold) | yes (threshold) | threshold | |
| Adaptive threshold | done | - | yes (-adaptive-threshold) | threshold | Mean or Gaussian |
| Otsu threshold | done | - | - | threshold | Auto-compute optimal level |
| Triangle threshold | done | - | - | threshold | Auto-compute via triangle method |
| **Alpha** | | | | | |
| Premultiply/unpremultiply | done | yes (premultiply) | - | alpha | |
| Add/remove alpha | done | yes (bandjoin/extract) | - | alpha | Registered as mappers |
| Flatten | done | yes (flatten) | yes (flatten) | alpha | Composite over solid bg |
| Mask apply | done | - | - | alpha | Apply alpha mask |
| Blend (over) | done | yes (composite) | yes (composite) | alpha | Porter-Duff over at offset |
| Blend if | done | - | - | alpha | Conditional blending |
| **Morphology** | | | | | |
| Erode/dilate/open/close | done | yes (morph) | yes (morphology) | morphology | |
| Gradient/tophat/blackhat | done | yes (morph) | yes (morphology) | morphology | |
| **Generators** | | | | | |
| Perlin noise | done | yes (gaussnoise) | yes (+noise) | generator | |
| Simplex noise | done | - | - | generator | |
| Gradient linear | done | yes (identity) | yes (gradient:) | generator | |
| Gradient radial | done | - | yes (radial-gradient:) | generator | |
| Checkerboard | done | - | yes (pattern:checkerboard) | generator | |
| Plasma | done | - | yes (plasma:) | generator | |
| **Effects / Artistic** | | | | | |
| Pixelate | done | - | yes (scale) | effect | IM parity: EXACT (MAE 0.00) |
| Halftone | done | - | - | effect | CMYK dot screen |
| Emboss | done | - | yes (emboss) | effect | Directional kernel |
| Oil paint | done | - | yes (paint) | effect | Neighborhood mode filter |
| Charcoal | done | - | yes (charcoal) | effect | Sobel + blur + invert (**outputs Gray8 — see §Known Issues**) |
| **Distortion** | | | | | |
| Swirl | done | - | yes (swirl) | distortion | IM parity: EXACT (MAE 0.001) |
| Spherize | done | - | - | distortion | Bulge/pinch |
| Barrel distortion | done | - | yes (distort Barrel) | distortion | IM parity: CLOSE (MAE 0.28) |
| Polar/depolar | done | - | yes (distort Polar) | distortion | |
| Wave | done | - | yes (wave) | distortion | Sinusoidal displacement |
| Ripple | done | - | - | distortion | Concentric wave displacement |
| **Grading** | | | | | |
| ASC CDL | done | - | - | grading | Industry-standard color correction |
| Lift/gamma/gain | done | - | - | grading | Professional 3-way grading |
| Split toning | done | - | - | grading | Highlight/shadow tinting |
| Curves (master/R/G/B) | done | - | - | grading | Spline-based tone curves, 4 channels |
| Film grain | done | - | - | effect | Procedural grain overlay |
| **Tonemapping** | | | | | |
| Tonemap Reinhard | done | - | - | tonemapping | HDR to SDR |
| Tonemap Drago | done | - | - | tonemapping | HDR to SDR |
| Tonemap Filmic | done | - | - | tonemapping | HDR to SDR |
| **Quantization** | | | | | |
| Quantize (median cut) | done | - | yes (quantize) | color | Color reduction |
| Dither (Floyd-Steinberg) | done | - | yes (dither) | color | Error diffusion |
| Dither (ordered/Bayer) | done | - | yes (ordered-dither) | color | |
| **Content-Aware** | | | | | |
| Smart crop | done | yes (smartcrop) | - | transform | Content-aware crop |
| Seam carve (width/height) | done | - | yes (liquid-rescale) | transform | |
| **3D Color LUT** | | | | | |
| Apply .cube LUT | done | - | yes (clut) | color | |
| Apply Hald LUT | done | - | yes (hald-clut) | color | |
| **Drawing** | | | | | |
| Draw line | done | yes (draw_line) | yes | draw | With width parameter |
| Draw rect | done | yes (draw_rect) | yes | draw | Fill + stroke |
| Draw circle | done | yes (draw_circle) | yes | draw | Fill + stroke |
| Draw text (bitmap) | done | - | - | draw | 8x16 bitmap font |
| Draw text (TTF) | done | - | yes (annotate) | draw | TrueType font rendering |
| Flood fill | done | yes (draw_flood) | yes (floodfill) | draw | |
| **Compositing** | | | | | |
| Blend modes (19) | done | yes (11 modes) | yes (40+ modes) | compositing | Multiply, Screen, Overlay, Darken, Lighten, SoftLight, HardLight, Difference, Exclusion, ColorDodge, ColorBurn, VividLight, LinearDodge, LinearBurn, LinearLight, PinLight, HardMix, Subtract, Divide |

**Total: 128 registered filters + 9 other registrations (generators, compositors, mappers, encoders, decoders) = 137 registered operations.**

### 5b. Known Issues

| Filter | Issue | Impact | Fix |
|--------|-------|--------|-----|
| **charcoal** | Outputs Gray8 but registered as `#[register_filter]` not `#[register_mapper]` | Pipeline nodes don't update format info → downstream filters receive wrong bpp | Re-register as mapper or auto-convert to RGB |
| **sobel, scharr, laplacian, canny** | Same Gray8 output issue as charcoal | Same pipeline format mismatch | Same fix needed |
| **grayscale** | Pipeline test explicitly `#[ignore]`d for this reason | Confirmed bug | Needs `#[register_mapper]` |

### 5c. Truly Missing Filters

No major filter gaps remain. All standard ImageMagick spatial, effect, and
distortion filters now have rasmcore equivalents with reference parity tests.

**Summary:** 0 truly missing filters. 128 registered filters covering all categories.

---

## 6. Color Management

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| sRGB | done | yes | yes | - | Default color space |
| ICC profile extract (JPEG) | done | yes | yes | - | color.rs |
| ICC profile extract (PNG) | done | yes | yes | - | color.rs |
| ICC to sRGB transform | done | yes (icc_transform) | yes | - | icc_to_srgb() |
| Lab (CIELAB) | done | yes | yes | - | rgb_to_lab / lab_to_rgb + image-level |
| Oklab/Oklch | done | yes | - | - | rgb_to_oklab / oklab_to_rgb in color_spaces.rs |
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

**Summary:** 17/21 color operations. Missing P1: full CMYK pipeline. Oklab/Oklch and Delta E 2000 are implemented.

---

## 7. Compositing & Blending

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Alpha composite (over) | done | yes | yes | - | Porter-Duff over (composite.rs) |
| Blend modes (19 modes) | done | yes (11 modes) | yes (40+ modes) | - | Multiply, Screen, Overlay, Darken, Lighten, SoftLight, HardLight, Difference, Exclusion, ColorDodge, ColorBurn, VividLight, LinearDodge, LinearBurn, LinearLight, PinLight, HardMix, Subtract, Divide |
| Flatten (alpha over bg) | done | yes | yes | - | flatten() with bg_r/g/b |
| Premultiply/unpremultiply | done | yes | - | - | Registered filters |
| Image concatenation | done | yes (arrayjoin) | yes (append) | - | concat_horizontal, concat_vertical, concat_grid |
| Montage/contact sheet | - | - | yes (montage) | P3 | |

**Summary:** 5/6 compositing ops. rasmcore has 19 blend modes vs libvips 11.

---

## 8. Drawing & Text

| Operation | rasmcore | libvips | ImageMagick | Priority | Notes |
|-----------|----------|---------|-------------|----------|-------|
| Draw line | done | yes (draw_line) | yes | - | Registered filter with width param |
| Draw rectangle | done | yes (draw_rect) | yes | - | Registered filter with fill + stroke |
| Draw circle | done | yes (draw_circle) | yes | - | Registered filter with fill + stroke |
| Text rendering (bitmap) | done | - | - | - | 8x16 bitmap font |
| Text rendering (TTF) | done | - | yes (annotate) | - | TrueType font rendering via draw_text_ttf |
| Flood fill | done | yes (draw_flood) | yes (floodfill) | - | Registered filter |

**Summary:** 6/6 drawing ops done. TTF text rendering available.

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

## Priority Summary (Updated 2026-03-31)

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
6. **Fix Gray8-output filters in pipeline** — charcoal, sobel, scharr, canny, grayscale need mapper registration or auto-convert

### Completed since last update
- ~~Register ~30 unregistered filters~~ **done** — 128 registered (was 49)
- ~~Draw primitives~~ **done** — line, rect, circle, text (bitmap + TTF), flood_fill
- ~~Image concatenation~~ **done** — horizontal, vertical, grid
- ~~Artistic effects~~ **done** — solarize, emboss, oil_paint, charcoal + pixelate, halftone, swirl, spherize, barrel, wave, ripple, polar/depolar
- ~~Oklab/Oklch~~ **done** — in color_spaces.rs
- ~~Delta E 2000~~ **done** — in color_spaces.rs
- ~~Hald CLUT~~ **done** — apply_hald_lut registered filter

### P2 — Medium
7. Lanczos2, Mitchell-Netravali resize filters
8. Linear sRGB / Display P3 full pipeline
9. Thread pool for parallel tile processing
10. Polygon/arc/pie drawing primitives

### P3 — Low / Niche
11. Shear transform
12. Camera RAW (blocked by no pure-Rust decoder)
13. PSD, DPX, CIN, DICOM

---

## 12. Cross-Ecosystem Comparison — Libraries

### vs sharp (Node.js / libvips)

| Category | rasmcore | sharp | Advantage |
|----------|----------|-------|-----------|
| Codec count | 19R/17W | 12R/12W | **rasmcore** — QOI, FITS, DDS, DNG, JP2, EXR, TGA, ICO, PNM, HDR |
| Blur variants | 8 (gaussian, bokeh, motion, zoom, spin, bilateral, tilt-shift, lens) | 1 (Gaussian) | **rasmcore** |
| Sharpen | unsharp mask | fast + accurate modes | Parity |
| Color adjustments | 20+ (brightness, contrast, gamma, levels, sigmoid, hue, sat, vibrance, sepia, colorize, photo filter, channel mixer, gradient map, modulate, selective color, white balance ×2, dodge, burn, clarity) | 8 (normalize, CLAHE, gamma, negate, linear, modulate, tint, recomb) | **rasmcore** |
| Edge detection | 5 (Sobel, Scharr, Laplacian, Canny, Hough) | 0 | **rasmcore** |
| Morphology | 7 ops | erode, dilate | **rasmcore** |
| Color grading | ASC CDL, LGG, split toning, curves (4ch), 3 tonemappers, film grain | 0 | **rasmcore** |
| Blend modes | 19 | 20+ via libvips composite | Parity |
| Content-aware | smart crop, seam carve ×2, inpainting | smartcrop (attention/entropy) | **rasmcore** |
| 3D LUT | .cube + Hald + compose | 0 | **rasmcore** |
| Drawing | line, rect, circle, text (bitmap+TTF), flood fill | 0 | **rasmcore** |
| Distortion | 7 (swirl, spherize, barrel, polar, depolar, wave, ripple) | 0 | **rasmcore** |
| Artistic | emboss, oil paint, charcoal, pixelate, halftone, solarize | 0 | **rasmcore** |
| Pipeline streaming | demand-driven tile cache | demand-driven via libvips | Parity |
| SIMD | libblur + LLVM auto-vec + WASM128 | Highway via libvips | **sharp** (more comprehensive dispatch) |
| Async I/O | no | Node.js streams | **sharp** |
| WASM support | **first-class** | no | **rasmcore** |
| Fluent API | pipeline DAG | `sharp().resize().blur().toFile()` | **sharp** (more ergonomic chaining) |

### vs Pillow (Python)

| Category | rasmcore | Pillow | Advantage |
|----------|----------|--------|-----------|
| Codec count | 19R/17W | ~30R/~20W | **Pillow** (more legacy formats via plugins) |
| Filters | 128 registered | ~15 (BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EMBOSS, FIND_EDGES, SHARPEN, SMOOTH, GaussianBlur, BoxBlur, MedianFilter, Min/Max/Mode, UnsharpMask) | **rasmcore** |
| Color adjustments | 20+ | 4 (Brightness, Contrast, Color, Sharpness via ImageEnhance) + ImageOps | **rasmcore** |
| 3D LUT | .cube + Hald + compose + absorb 1D | Color3DLUT (basic) | **rasmcore** |
| Drawing | line, rect, circle, text (bitmap+TTF), flood fill | line, rect, ellipse, polygon, arc, chord, pie, text (FreeType) | **Pillow** (more shape primitives) |
| Color grading | ASC CDL, LGG, split toning, curves, tonemapping, film grain | 0 | **rasmcore** |
| Quantization | median cut, Floyd-Steinberg, ordered | median cut, octree, libimagequant | **Pillow** (libimagequant) |
| SIMD | libblur + WASM128 | None (Pillow-SIMD fork: AVX2) | Tie (different targets) |
| Metadata | EXIF R/W, XMP, IPTC, ICC | EXIF R/W, ICC, XMP | Parity |
| Ecosystem | Rust/WASM | Massive Python ecosystem (scikit-image, OpenCV, numpy interop) | **Pillow** |
| Font rendering | bitmap 8x16 + TTF | FreeType with full hinting | **Pillow** (more mature) |
| WASM | **first-class** | no | **rasmcore** |

### vs image-rs (Rust)

| Category | rasmcore | image-rs | Advantage |
|----------|----------|----------|-----------|
| Codec count | 19R/17W | 15R/14W (no HEIC, JXL, JP2, DNG, FITS) | **rasmcore** |
| WebP write | lossy + lossless | lossless only | **rasmcore** |
| Filters | 128 registered | ~10 (blur, unsharpen, brighten, contrast, huerotate, grayscale, invert, crop, resize, flip, rotate) | **rasmcore** (12x more) |
| Color grading | Full suite | 0 | **rasmcore** |
| Edge detection | 5 | 0 | **rasmcore** |
| Content-aware | smart crop, seam carve, inpainting | 0 | **rasmcore** |
| Metadata | EXIF R/W, XMP, IPTC, ICC | ICC only (EXIF via external crate) | **rasmcore** |
| Pipeline | demand-driven DAG | immediate (full image in memory) | **rasmcore** |
| API simplicity | function calls (same directness) + pipeline DAG | `img.resize().blur().save()` | Parity for basic use; **rasmcore ahead** with pipeline option |
| Community | new | 220M+ downloads | **image-rs** |
| WASM | **first-class** | possible but not primary | **rasmcore** |
| Plugin system | `#[register_filter]` + inventory | decoder/encoder registration | **rasmcore** |

---

## 13. Cross-Ecosystem Comparison — Professional Tools

### vs Adobe Photoshop

| Category | rasmcore | Photoshop | Notes |
|----------|----------|-----------|-------|
| **Adjustment layers** | | | |
| Brightness/Contrast | done | done | |
| Levels | done | done | 5-param (black/white in/out + gamma) |
| Curves | done (master + R/G/B) | done (RGB + individual) | Parity |
| Exposure | - | done | Missing — logarithmic exposure control |
| Vibrance | done | done | |
| Hue/Saturation | done (hue_rotate + saturate) | done | |
| Color Balance | - | done (shadows/mids/highlights) | **Missing** — lift/gamma/gain is similar but not identical |
| Black & White | done (grayscale) | done (with per-channel weights) | PS has more control |
| Photo Filter | done | done | |
| Channel Mixer | done | done | |
| Color Lookup (3D LUT) | done (.cube + Hald) | done | Parity |
| Invert | done | done | |
| Posterize | done | done | |
| Threshold | done (binary + Otsu + triangle) | done | **rasmcore ahead** (auto thresholds) |
| Gradient Map | done | done | |
| Selective Color | done | done | |
| **Score** | **14/16** | **16/16** | Missing: Exposure, Color Balance |
| **Blur types** | | | |
| Gaussian | done | done | |
| Motion | done | done | |
| Radial/Zoom | done (zoom_blur) | done | |
| Spin | done (spin_blur) | done | |
| Lens | done (lens_blur) | done | |
| Box | - | done | Missing |
| Shape | - | done | Missing |
| Smart/Surface/Bilateral | done (bilateral + guided) | done | **rasmcore ahead** (bilateral + guided) |
| Bokeh | done | - | **rasmcore unique** (disc/hex/oct/star) |
| Tilt-Shift | done | done | |
| Average | - | done | Missing (trivial) |
| **Score** | **8/11 PS types + 1 unique** | **11/11** | Missing: Box, Shape, Average |
| **Sharpen** | | | |
| Unsharp Mask | done | done | |
| Smart Sharpen | - | done | Missing (edge-aware) |
| High Pass | done (frequency_high) | done | |
| **Distort** | | | |
| Spherize | done | done | |
| Twirl/Swirl | done | done | |
| Wave | done | done | |
| Ripple | done | done | |
| Polar Coordinates | done (polar/depolar) | done | |
| Displace | done (displacement_map) | done | |
| Pinch | done (spherize negative) | done | |
| Shear | - | done | Missing (can do via affine) |
| Glass | - | done | Missing |
| Ocean Ripple | - | done | Missing (similar to ripple) |
| ZigZag | - | done | Missing |
| Diffuse Glow | - | done | Missing |
| **Score** | **7/12** | **12/12** | |
| **Artistic / Stylize** | | | |
| Oil Paint | done | done | |
| Emboss | done | done | |
| Solarize | done | done | |
| Find Edges (Sobel) | done | done | |
| Film Grain | done | done | |
| Halftone | done | done (Color Halftone) | |
| Charcoal | done | done (Sketch > Charcoal) | |
| Pixelate/Mosaic | done | done | |
| Watercolor | - | done | Missing |
| Colored Pencil | - | done | Missing |
| Palette Knife | - | done | Missing |
| Neon Glow / Glowing Edges | - | done | Missing |
| Wind | - | done | Missing |
| **Score** | **8/13** | **13/13** + 40 more (Artistic/Brush/Sketch/Texture galleries) | PS has ~50 more artistic filters |
| **Blend modes** | 19 | 27 (+4 special) | PS has 8 more: Dissolve, Darker/Lighter Color, Hue, Saturation, Color, Luminosity, Behind, Clear |
| **Content-aware** | smart crop, seam carve, inpainting (Telea/NS) | Content-Aware Fill/Move/Scale/Patch, Remove Tool, Generative Fill/Expand | **PS ahead** — AI-powered features |
| **Color management** | ICC read, Lab/Oklab/ProPhoto/AdobeRGB, Bradford adapt | Full ICC pipeline, CMYK editing, soft proofing, 4 rendering intents | **PS ahead** — full ICC transform + CMYK editing |
| **Bit depth** | 8-bit + 16-bit | 8/16/32-bit (HDR) | PS has 32-bit float editing |

### vs GIMP

| Category | rasmcore | GIMP | Notes |
|----------|----------|------|-------|
| **Color adjustments** | 20+ | 54+ (Colors menu) | **GIMP ahead** — more fine-grained color tools |
| **Blur** | 8 | 12 | **GIMP ahead** — Mean Curvature, Focus, Variable, Tileable |
| **Edge detection** | 5 (Sobel, Scharr, Laplacian, Canny, Hough) | 6 (DoG, Edge, Laplace, Neon, Sobel, Image Gradient) | Parity (different sets) |
| **Tone mapping** | 3 (Reinhard, Drago, Filmic) | 4 (Fattal, Mantiuk, Reinhard, Stress) | Parity |
| **Noise generation** | 3 (Perlin, Simplex, Plasma) + film grain | 7 (CIE LCh, HSV, Hurl, Pick, RGB, Slur, Spread) | **GIMP ahead** |
| **Distortion** | 7 | 19 | **GIMP ahead** — Kaleidoscope, Mosaic, Newsprint, etc. |
| **Artistic** | 6 (emboss, oil, charcoal, pixelate, halftone, solarize) | 17+ | **GIMP ahead** — Cartoon, Cubism, Glass Tile, Van Gogh LIC, etc. |
| **Blend modes** | 19 | ~38 (incl. LCh modes, Grain Extract/Merge) | **GIMP ahead** |
| **Color grading** | ASC CDL, LGG, split toning, curves, film grain | 0 (no dedicated grading) | **rasmcore ahead** |
| **3D LUT** | .cube + Hald + compose | 0 | **rasmcore ahead** |
| **Content-aware** | smart crop, seam carve, inpainting | 0 | **rasmcore ahead** |
| **Morphology** | 7 ops | erode, dilate | **rasmcore ahead** |
| **Image metrics** | MAE, RMSE, PSNR, SSIM, Delta E | 0 | **rasmcore ahead** |
| **WASM** | first-class | no | **rasmcore** |
| **Pipeline** | demand-driven DAG, plugin system | GEGL-based pipeline | Both sophisticated |
| **Scripting** | Rust API + WIT component model | Script-Fu + Python-Fu | **GIMP** (more mature) |
| **Formats** | 19R/17W | 73+ extensions | **GIMP ahead** (legacy format breadth) |

### vs DaVinci Resolve (Color Page)

| Category | rasmcore | DaVinci Resolve | Notes |
|----------|----------|-----------------|-------|
| **Primary grading** | | | |
| Lift/Gamma/Gain | done | done (+ Log wheels, HDR wheels) | **Resolve ahead** — Log + HDR zone wheels |
| ASC CDL | done | done (import/export) | Parity |
| Curves | done (master + R/G/B) | done (Custom + 5 Hue/Sat/Lum curves) | **Resolve ahead** — Hue vs X curves |
| Split toning | done | done (via wheels) | Parity |
| Film grain | done | done (ResolveFX) | Parity |
| 3D LUT | done (.cube + Hald + compose) | done (1D + 3D, browse, generate) | Parity |
| Color Warper | - | done | **Missing** — mesh-based hue/sat/lum warping |
| **Color science** | | | |
| Lab/Oklab/ProPhoto/AdobeRGB | done | ACES, DaVinci Wide Gamut, RCM | Different approaches |
| White balance | done (gray world + temperature) | done (Color Match) | Parity |
| Gamut mapping/limiting | - | done | **Missing** |
| HDR metadata (Dolby Vision, HDR10+) | - | done (Studio) | **Missing** — video domain |
| **Blur** | 8 | 7 (Box, Directional, Gaussian, Lens, Mosaic, Radial, Zoom) | Parity |
| **Noise reduction** | NLM, bilateral, guided, median | Temporal + Spatial NR (multi-frame, Studio) | **Resolve ahead** (temporal NR) |
| **Sharpening** | unsharp mask | 3 types (Studio) | Parity |
| **Artistic/Stylize** | 6 effects | 13 ResolveFX Stylize | **Resolve ahead** |
| **Dehaze** | done | done (Studio) | Parity |
| **Scopes** | histogram, statistics | Waveform, Vectorscope, Parade, Histogram, CIE | **Resolve ahead** |
| **Node pipeline** | DAG with spatial cache | Serial/Parallel/Layer/Splitter nodes | **Resolve ahead** (more node types) |
| **Real-time playback** | not applicable | yes (GPU accelerated) | Different domain |
| **WASM** | first-class | no | **rasmcore** |

**Key takeaway vs Resolve:** rasmcore already covers the core color grading operations (CDL, LGG, curves, LUT, split toning, film grain, dehaze) that make up 80%+ of a typical color grade. The gaps are in advanced Resolve-specific features (Color Warper, Hue vs X curves, HDR metadata, temporal NR) and video-domain features (real-time playback, tracking, scopes).

---

## rasmcore Unique Advantages

| Advantage | Detail |
|-----------|--------|
| **Pure Rust / WASM** | Compiles to wasm32-wasip2, runs in browsers and edge compute |
| **No C dependencies** | Core codecs are pure Rust — no FFI vulnerabilities |
| **QOI format** | Native support — neither IM nor vips has it |
| **DDS read+write** | Both IM and vips have limited DDS support |
| **19 blend modes** | More than libvips (11), validated against vips + ImageMagick |
| **Professional grading** | ASC CDL, lift/gamma/gain, curves — uncommon in image libraries |
| **Content-aware ops** | Smart crop + seam carve + selective color + inpainting |
| **Color science** | Lab, Oklab, ProPhoto, Adobe RGB, LCH, Luv, Delta E (76/94/2000) |
| **3D Color LUTs** | .cube + Hald, compose, absorb 1D pre/post |
| **Component Model** | WIT interfaces enable cross-language composability |
| **137 registered operations** | 128 filters + generators + compositors + mappers + encoders + decoders |
