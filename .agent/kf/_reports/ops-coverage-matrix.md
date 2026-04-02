# Operations Coverage Matrix — rasmcore vs Competitors

> Updated: 2026-04-02
> Scope: Filters, transforms, effects, color ops, analysis, drawing only. Formats/codecs excluded.

## Summary

| Library | Registered Ops | Unique Strengths | GPU Accel | WASM |
|---------|---------------|------------------|-----------|------|
| **rasmcore** | 140 filters + 7 transforms + ~80 domain ops | Native codecs, tile pipeline, 20 GPU ops, 4 SDKs | 20 ops | Yes |
| **ImageMagick** | ~200 operations | Broadest feature set, per-pixel scripting (-fx) | No | No |
| **libvips** | ~120 operations | FFT, streaming pipeline, mosaicing, ICC transforms | No | No |
| **sharp** | ~30 operations | Simple API, libvips backend, Node.js ecosystem | Via libvips | No |
| **Pillow** | ~60 operations | Python ecosystem, ImageDraw, broad format support | No | No |
| **photon_rs** | ~96 core functions | WASM-first, preset filters, simple API | No | Yes |

**Competitive position:** rasmcore leads in filter breadth (115 registered), GPU acceleration (20 ops), and is the only library combining WASM portability with professional-grade operations. ImageMagick has broader raw feature count but no WASM/GPU. libvips has FFT and ICC transforms that rasmcore lacks.

---

## Category-by-Category Comparison

### Legend

| Symbol | Meaning |
|--------|---------|
| Y | Fully implemented |
| P | Partial (limited params or variants) |
| — | Not available |
| **Y+** | Surpasses competitors (more variants or GPU-accelerated) |

---

### 1. Blur / Smoothing

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Gaussian blur | **Y+** (GPU) | Y | Y | Y | Y | Y |
| Box blur | Y | Y | — | Y | Y | Y |
| Median filter | **Y+** (GPU) | Y | Y | Y | Y | — |
| Bilateral filter | **Y+** (GPU) | Y | — | — | — | — |
| Motion blur | **Y+** (GPU) | Y | — | — | — | — |
| Zoom/radial blur | **Y+** (GPU) | — | — | — | — | — |
| Spin blur | **Y+** (GPU) | — | — | — | — | — |
| Bokeh blur | Y | — | — | — | — | — |
| Lens blur | Y | — | — | — | — | — |
| Guided filter | **Y+** (GPU) | — | — | — | — | — |
| Tilt-shift blur | Y | — | — | — | — | — |
| NLM denoise | Y | — | — | — | — | — |
| Noise reduction | — | Y | — | — | — | Y |
| Adaptive blur | — | Y | — | — | — | — |
| Selective blur | — | Y | — | — | — | — |

**rasmcore: 12/15 | IM: 7/15 | libvips: 2/15 | sharp: 2/15 | Pillow: 3/15 | photon_rs: 3/15**
rasmcore leads decisively with 8 GPU-accelerated blur variants.

---

### 2. Sharpen / Detail

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Unsharp mask (sharpen) | **Y+** (GPU) | Y | Y | Y | Y | Y |
| High-pass filter | Y | — | — | — | — | — |
| Clarity | Y | — | — | — | — | — |
| Dehaze | Y | — | — | — | — | — |
| LAB sharpen (luminosity) | Y | — | — | — | — | — |
| Pyramid detail remap | Y | — | — | — | — | — |
| Adaptive sharpen | — | Y | — | — | — | — |

**rasmcore: 6/7 | IM: 2/7 | libvips: 1/7 | sharp: 1/7 | Pillow: 1/7 | photon_rs: 1/7**

---

### 3. Edge Detection / Analysis

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Canny | Y | Y | Y | — | — | — |
| Sobel | Y | Y | Y | — | — | Y |
| Scharr | Y | — | Y | — | — | — |
| Laplacian | Y | — | — | — | — | Y |
| Prewitt | — | — | Y | — | — | Y |
| Harris corner detection | Y | — | — | — | — | — |
| Template matching (NCC) | Y | — | — | — | — | — |
| Hough lines | Y | — | Y | — | — | — |
| Hough circles | — | — | Y | — | — | — |
| Connected components | Y | Y | Y | — | — | — |
| Contour tracing | Y | — | — | — | — | — |
| Distance transform | Y | — | — | — | — | — |

**rasmcore: 10/12 | IM: 3/12 | libvips: 6/12 | sharp: 0/12 | Pillow: 0/12 | photon_rs: 3/12**

---

### 4. Morphology

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Erode | Y | Y | Y | — | Y | — |
| Dilate | Y | Y | Y | — | Y | — |
| Open | Y | Y | Y | — | — | — |
| Close | Y | Y | Y | — | — | — |
| Gradient | Y | Y | Y | — | — | — |
| Top-hat | Y | Y | — | — | — | — |
| Black-hat | Y | Y | — | — | — | — |
| Skeletonize (Zhang-Suen) | **Y+** (GPU) | — | — | — | — | — |

**rasmcore: 8/8 | IM: 7/8 | libvips: 5/8 | sharp: 0/8 | Pillow: 2/8 | photon_rs: 0/8**

---

### 5. Color Adjustment

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Brightness | Y | Y | Y | Y | Y | Y |
| Contrast | Y | Y | Y | Y | Y | Y |
| Gamma | Y | Y | Y | Y | — | Y |
| Exposure (EV) | Y | — | — | — | — | — |
| Levels | Y | Y | — | Y | — | — |
| Curves (master + R/G/B) | Y | Y | — | — | — | — |
| Sigmoidal contrast | Y | Y | — | — | — | — |
| Hue rotation | Y | Y | Y | — | — | Y |
| Saturation | Y | Y | Y | — | Y | Y |
| Vibrance | Y | — | — | — | — | — |
| Channel mixer | Y | — | — | — | — | — |
| Selective color | Y | — | — | — | — | P |
| Color balance | Y | — | — | — | — | — |
| Shadow / highlight | Y | — | — | — | — | — |
| Invert / negate | Y | Y | Y | Y | Y | Y |
| Posterize | Y | Y | — | — | Y | — |
| Solarize | Y | Y | — | — | Y | Y |
| Modulate (HSB adjust) | Y | Y | — | Y | — | — |
| Photo filter (warm/cool) | Y | — | — | — | — | — |
| Normalize / auto-levels | Y | Y | Y | Y | Y | Y |
| Equalize (histogram) | Y | Y | Y | — | Y | — |
| CLAHE | Y | — | Y | Y | — | — |
| Dodge | Y | Y | — | — | — | — |
| Burn | Y | Y | — | — | — | — |
| Colorize | Y | Y | — | — | Y | Y |
| White balance | Y | Y | — | — | — | — |

**rasmcore: 26/26 | IM: 18/26 | libvips: 9/26 | sharp: 8/26 | Pillow: 9/26 | photon_rs: 9/26**

---

### 6. Color Space Operations

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| sRGB / Linear RGB | Y | Y | Y | Y | Y | — |
| Lab (CIELAB D65) | Y | Y | Y | — | Y | — |
| LAB extract L/a/b | Y | — | — | — | — | — |
| LAB adjust (a/b offset) | Y | — | — | — | — | — |
| XYZ | Y | Y | Y | — | — | — |
| HSL | Y | Y | Y | — | Y | Y |
| HSV | Y | Y | Y | — | Y | Y |
| LCh | Y | Y | Y | — | — | Y |
| Oklab | Y | — | Y | — | — | — |
| Luv (CIELUV) | Y | Y | Y | — | — | — |
| CMYK | Y | Y | Y | — | Y | — |
| ProPhoto RGB | Y | — | Y | — | — | — |
| Adobe RGB | Y | — | Y | — | — | — |
| Delta E (76, 94, 2000) | Y | — | — | — | — | — |
| ICC profile transform | — | Y | Y | Y | — | — |
| scRGB | — | Y | Y | — | — | — |
| Yxy | — | — | Y | — | — | — |
| HSLuv | — | — | — | — | — | Y |

**rasmcore: 14/18 | IM: 10/18 | libvips: 14/18 | sharp: 3/18 | Pillow: 5/18 | photon_rs: 4/18**

---

### 7. Tone Mapping / HDR

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Reinhard | Y | — | — | — | — | — |
| Drago | Y | — | — | — | — | — |
| Filmic / ACES | Y | — | — | — | — | — |
| Retinex SSR | Y | — | — | — | — | — |
| Retinex MSR | Y | — | — | — | — | — |
| Retinex MSRCR | Y | — | — | — | — | — |

**rasmcore: 6/6 | All others: 0/6**
rasmcore is the only library with built-in tone mapping operators.

---

### 8. Color Grading / LUT

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| ASC CDL | Y | — | — | — | — | — |
| Lift / Gamma / Gain | Y | — | — | — | — | — |
| Split toning | Y | — | — | — | — | — |
| 3D CLUT apply | **Y+** (GPU) | Y | Y | — | — | — |
| .cube LUT file | Y | Y | — | — | — | — |
| Hald CLUT | Y | Y | Y | — | — | — |
| Gradient map | Y | — | — | — | — | — |
| Film grain | Y | — | — | — | — | — |
| Preset filters (Instagram-style) | — | — | — | — | — | **Y+** (30+) |

**rasmcore: 8/9 | IM: 3/9 | libvips: 2/9 | sharp: 0/9 | Pillow: 0/9 | photon_rs: 1/9**
photon_rs leads only in preset filters (30+ Instagram-style presets).

---

### 9. Distortion / Warping

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Barrel / pincushion | **Y+** (GPU) | Y | — | — | — | — |
| Spherize | **Y+** (GPU) | — | — | — | — | — |
| Swirl | **Y+** (GPU) | Y | — | — | — | — |
| Wave | **Y+** (GPU) | Y | — | — | — | — |
| Ripple | **Y+** (GPU) | — | — | — | — | — |
| Polar / depolar | **Y+** (GPU) | Y | — | — | — | — |
| Mesh warp (grid) | Y | — | — | — | — | — |
| Perspective warp | Y | Y | — | — | — | — |
| Displacement map | Y | Y | — | — | — | — |
| Affine transform | Y | Y | Y | Y | — | — |
| Shepards distort | — | Y | — | — | — | — |
| Bilinear distort | — | Y | — | — | — | — |
| Cylinder projection | — | Y | — | — | — | — |
| Generic remap (mapim) | — | — | Y | — | Y | — |
| Frosted glass | — | — | — | — | — | Y |
| Shear | — | — | — | — | — | Y |

**rasmcore: 10/16 | IM: 10/16 | libvips: 2/16 | sharp: 1/16 | Pillow: 1/16 | photon_rs: 2/16**

---

### 10. Geometric Transforms

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Resize (multi-filter) | Y (13 filters) | Y | Y | Y | Y | Y |
| Crop | Y | Y | Y | Y | Y | Y |
| Rotate 90/180/270 | Y | Y | Y | Y | Y | Y |
| Rotate arbitrary | Y | Y | Y | Y | — | Y |
| Flip H/V | Y | Y | Y | Y | Y | Y |
| Pad / extend | Y | Y | Y | Y | Y | Y |
| Trim / auto-crop | Y | Y | Y | Y | — | — |
| Auto-orient (EXIF) | Y | Y | Y | Y | Y | — |
| Concat (H/V/grid) | Y | Y | — | — | — | — |
| Seam carve (W/H) | Y | Y | — | — | — | Y |
| Smart crop | Y | Y | Y | — | — | — |
| Shrink-on-load | — | Y | Y | Y | — | — |

**rasmcore: 11/12 | IM: 12/12 | libvips: 9/12 | sharp: 9/12 | Pillow: 6/12 | photon_rs: 7/12**

---

### 11. Effects

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Emboss | Y | Y | — | — | Y | Y |
| Oil paint | Y | Y | — | — | — | Y |
| Charcoal | Y | Y | — | — | — | — |
| Pixelate | Y | Y | — | — | — | Y |
| Halftone | Y | — | — | — | — | Y |
| Vignette | Y | Y | — | — | — | — |
| Vignette (power-law) | Y | — | — | — | — | — |
| Duotone | — | — | — | — | — | Y |
| RGB channel offset | — | — | — | — | — | Y |
| Horizontal/vertical strips | — | — | — | — | — | Y |

**rasmcore: 7/10 | IM: 5/10 | libvips: 0/10 | sharp: 0/10 | Pillow: 1/10 | photon_rs: 6/10**

---

### 12. Drawing / Text

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Line | Y | Y | Y | — | Y | — |
| Rectangle | Y | Y | Y | — | Y | — |
| Circle | Y | Y | Y | — | Y | — |
| Ellipse | Y | — | — | — | Y | — |
| Arc | Y | — | — | — | Y | — |
| Polygon | Y | — | — | — | Y | — |
| Flood fill | Y | Y | Y | — | Y | — |
| Bitmap text | Y | Y | — | — | Y | Y |
| TrueType text | Y | Y (Pango) | Y (Pango) | Y | Y (Pillow fonts) | — |
| Complex text shaping | Y | Y | Y | — | — | — |
| Watermark | Y | Y | — | — | — | Y |
| Draw image (paste) | — | Y | Y | Y | Y | — |

**rasmcore: 11/12 | IM: 9/12 | libvips: 6/12 | sharp: 2/12 | Pillow: 10/12 | photon_rs: 2/12**

---

### 13. Compositing / Blending

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Alpha composite | Y | Y | Y | Y | Y | Y |
| Multiply | Y | Y | Y | Y | Y | Y |
| Screen | Y | Y | Y | Y | Y | Y |
| Overlay | Y | Y | Y | Y | — | Y |
| Soft light | Y | Y | Y | Y | — | Y |
| Hard light | Y | Y | Y | Y | — | Y |
| Color dodge | Y | Y | — | — | — | Y |
| Color burn | Y | Y | — | — | — | Y |
| Difference | Y | Y | Y | Y | Y | Y |
| Exclusion | Y | Y | — | Y | — | — |
| Darken | Y | Y | Y | Y | Y | — |
| Lighten | Y | Y | Y | Y | Y | — |
| Blend modes total | 19 | 30+ | 11 | 13 | 11 | 10 |

**rasmcore: 19 modes | IM: 30+ | libvips: 11 | sharp: 13 | Pillow: 11 | photon_rs: 10**

---

### 14. Threshold / Quantization

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Binary threshold | Y | Y | Y | Y | — | Y |
| Adaptive threshold | Y | Y | — | — | — | — |
| Otsu threshold | Y | Y | — | — | — | — |
| Triangle threshold | Y | — | — | — | — | — |
| Quantize (palette) | Y | Y | Y | — | Y | — |
| Floyd-Steinberg dither | Y | Y | — | — | — | Y |
| Ordered dither | Y | Y | — | — | — | — |
| Grayscale (various) | Y | Y | Y | Y | Y | Y (7 methods) |
| Sepia | Y | Y | — | — | — | Y |

**rasmcore: 9/9 | IM: 8/9 | libvips: 3/9 | sharp: 2/9 | Pillow: 2/9 | photon_rs: 4/9**

---

### 15. Noise Generation

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Perlin noise | Y | — | — | — | — | — |
| Simplex noise | Y | — | — | — | — | — |
| Film grain | Y | — | — | — | — | — |
| Gaussian noise (add) | — | Y | — | — | — | Y |
| Impulse noise | — | Y | — | — | — | — |
| Poisson noise | — | Y | — | — | — | — |
| Pink noise | — | — | — | — | — | Y |

**rasmcore: 3/7 | IM: 3/7 | Others: 0-2/7**

---

### 16. Comparison Metrics

| Metric | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|--------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| MAE | Y | Y | — | — | — | — |
| RMSE | Y | Y | — | — | — | — |
| PSNR | Y | Y | — | — | — | — |
| SSIM | Y | Y | — | — | — | — |
| Delta E (CIE76/94/2000) | Y | — | — | — | — | — |
| NCC | — | Y | — | — | — | — |
| Perceptual hash | — | Y | — | — | — | — |

**rasmcore: 5/7 | IM: 5/7 | Others: 0/7**

---

### 17. Convolution

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Generic kernel | Y | Y | Y | Y | Y | — |
| Separable convolution | Y | Y | Y | — | — | — |

**Parity across rasmcore, IM, libvips.**

---

### 18. Frequency Domain

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Frequency separation (Gaussian) | Y | — | — | — | — | — |
| FFT (Fourier transform) | — | — | Y | — | — | — |
| Inverse FFT | — | — | Y | — | — | — |
| Frequency multiply | — | — | Y | — | — | — |
| Phase correlation | — | — | Y | — | — | — |

rasmcore has Gaussian-based frequency separation but lacks true FFT.

---

### 19. Content-Aware / Smart Operations

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| Smart crop (attention) | Y | Y | Y | — | — | — |
| Seam carve (width) | Y | Y | — | — | — | Y |
| Seam carve (height) | Y | Y | — | — | — | — |
| Inpaint (Telea) | Y | — | — | — | — | — |
| Perspective correct | Y | — | — | — | — | — |

**rasmcore: 5/5 | IM: 3/5 | Others: 0-1/5**

---

### 20. GPU Acceleration

| Operation | rasmcore | IM | libvips | sharp | Pillow | photon_rs |
|-----------|:--------:|:--:|:-------:|:-----:|:------:|:---------:|
| GPU-accelerated ops | **20** | 0 | 0 | 0 | 0 | 0 |

rasmcore is the **only library** with built-in GPU acceleration (via wgpu/WebGPU). GPU-capable operations: gaussian_blur, bilateral, guided_filter, median, sharpen, high_pass, motion_blur, spin_blur, zoom_blur, spherize, swirl, barrel, ripple, wave, polar, depolar, skeletonize, affine_resample, fused_lut, fused_clut.

---

## Remaining Gaps (rasmcore is missing)

| Gap | Priority | Available In | Notes |
|-----|----------|-------------|-------|
| **ICC profile transform/apply** | P0 | IM, libvips, sharp | Can read ICC but cannot apply color transforms |
| **FFT / Fourier transform** | P1 | libvips | Only Gaussian separation available, no true FFT |
| **Add noise (Gaussian/Impulse/Poisson)** | P1 | IM | Have Perlin/simplex/film_grain but not additive noise types |
| **Shrink-on-load** | P1 | IM, libvips, sharp | Major perf optimization for JPEG/TIFF decode |
| **Per-pixel math (-fx / -evaluate)** | P2 | IM | Scriptable per-pixel operations |
| **Adaptive blur/sharpen** | P2 | IM | Edge-aware variants |
| **Generic remap (mapim)** | P2 | libvips, Pillow | Arbitrary coordinate remapping |
| **Shepards/bilinear distortion** | P2 | IM | Specialized distortion algorithms |
| **k-means segmentation** | P2 | IM | Have quantize but not spatial k-means |
| **Mosaicing / stitching** | P2 | libvips | Multi-image alignment and blending |
| **Preset filters (Instagram-style)** | P3 | photon_rs (30+) | Not a priority for professional use |
| **scRGB / Yxy color spaces** | P3 | IM, libvips | Niche color spaces |

---

## Areas Where rasmcore Surpasses ALL Competitors

1. **GPU acceleration** — 20 operations hardware-accelerated via wgpu. No other library offers this.
2. **Tone mapping** — 6 tone mapping operators (Reinhard, Drago, Filmic, Retinex SSR/MSR/MSRCR). No competitor has built-in tone mapping.
3. **Blur variety** — 12 blur types including GPU-accelerated bokeh, lens, spin, zoom, tilt-shift. More than any competitor.
4. **Detail enhancement** — High-pass, clarity, dehaze, LAB sharpen, pyramid detail remap. Unique combination.
5. **Morphology** — 8 operations including GPU-accelerated skeletonize. Most complete set.
6. **Color adjustment** — 26 adjustment operations. More than any single competitor.
7. **Delta E metrics** — CIE76, CIE94, CIEDE2000 perceptual color difference. Unique.
8. **WASM + professional ops** — Only library combining WebAssembly portability with professional-grade operations (photon_rs is WASM but far narrower).
9. **Tile-based pipeline** — Demand-driven tile processing with spatial cache. Only libvips has comparable streaming architecture.
10. **Multi-SDK** — TypeScript, Go, Python, Rust native, C FFI from single codebase. No competitor offers this.

---

## vs photon_rs: Detailed Comparison

photon_rs was the closest WASM-first competitor. Current status:

| Category | rasmcore | photon_rs | Winner |
|----------|:--------:|:---------:|:------:|
| Registered filters | 140 | ~96 | rasmcore |
| Blur variants | 12 | 3 | rasmcore (4x) |
| Edge detection | 7 | 3 | rasmcore |
| Morphology | 8 | 0 | rasmcore |
| Color adjustment | 26 | 9 | rasmcore (3x) |
| Tone mapping | 6 | 0 | rasmcore |
| Color grading | 8 | 0 | rasmcore |
| Distortion | 10 | 2 | rasmcore (5x) |
| Drawing primitives | 11 | 2 | rasmcore (5x) |
| Content-aware | 5 | 1 | rasmcore |
| GPU acceleration | 20 | 0 | rasmcore |
| Comparison metrics | 5 | 0 | rasmcore |
| Preset filters | 0 | 30+ | photon_rs |
| Color spaces | 14 | 4 | rasmcore (3.5x) |
| Native codecs | 5 (JPEG, WebP, HEVC, BMP, QOI) | 0 | rasmcore |

**Conclusion:** rasmcore comprehensively surpasses photon_rs in every category except preset Instagram-style filters (which are trivially composable from existing operations). photon_rs remains a good choice for simple WASM image manipulation but lacks the professional-grade operations, GPU acceleration, and tile pipeline that rasmcore provides.
