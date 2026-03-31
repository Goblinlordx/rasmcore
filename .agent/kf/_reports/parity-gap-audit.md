# Parity Gap Audit: rasmcore vs libvips, ImageMagick, sharp, Pillow

**Date:** 2026-03-31 (updated)
**Track:** parity-gap-audit_20260330094610Z

---

## 1. Format Support Matrix

| Format | rasmcore | libvips | ImageMagick | sharp | Pillow |
|--------|:--------:|:-------:|:-----------:|:-----:|:------:|
| **JPEG** | R/W | R/W | R/W | R/W | R/W |
| **PNG** | R/W | R/W | R/W | R/W | R/W |
| **APNG** | R/W | R/W | R/W | R/W | R/W |
| **GIF** | R/W | R/W | R/W | R/W | R/W |
| **WebP** | R/W | R/W | R/W | R/W | R/W |
| **TIFF** | R/W | R/W | R/W | R/W | R/W |
| **BMP** | R/W | — | R/W | — | R/W |
| **AVIF** | W | R/W | R/W | R/W | R/W |
| **HEIC/HEIF** | R/W | R/W | R/W | R/W | — |
| **JPEG XL** | R | R/W | R/W | — | — |
| **JP2 (JPEG 2000)** | R/W | R/W | R/W | — | R/W |
| **EXR (OpenEXR)** | R/W | R | R/W | — | — |
| **HDR (Radiance)** | R/W | R/W | R/W | — | — |
| **SVG** | R | R | R/W | R | — |
| **PDF** | — | R/W | R/W | R | W |
| **QOI** | R/W | — | R/W | — | R/W |
| **ICO** | R/W | — | R | — | R/W |
| **TGA** | R/W | — | R/W | — | R/W |
| **PNM/PPM** | R/W | R/W | R/W | — | R/W |
| **DDS** | R/W | — | R/W | — | R/W |
| **FITS** | R/W | R | R/W | — | R |
| **RAW/DNG** | — | R | R | R | — |
| **DICOM** | — | — | R | — | — |
| **PSD** | — | — | R | — | R |
| **DZI (Deep Zoom)** | — | W | — | — | — |

**rasmcore: 19 decode, 17 encode (137 registered operations)** | libvips: ~25 decode, ~20 encode | IM: 100+ | sharp: ~12 | Pillow: ~30 decode, ~20 encode

### Format Gaps (rasmcore is missing)

| Gap | Priority | Impact | Notes |
|-----|----------|--------|-------|
| **AVIF decode** | P0 | High — AVIF is standard web format, we can encode but not decode | Need rav1d or dav1d binding |
| **JPEG XL encode** | P1 | Medium — browser adoption still limited, but growing | jxl-oxide has encode support |
| **RAW/DNG decode** | P1 | Medium — important for photography workflows | dcraw/LibRaw binding or pure Rust |
| **Animated WebP encode** | P1 | Medium — GIF replacement use case | image-webp crate limitation |
| **PDF read** | P2 | Low — niche use case for image library | Would need poppler/mupdf binding |
| **PSD read** | P2 | Low — Photoshop interop | Complex format, limited demand |
| **SVG write** | P2 | Low — we render SVG to raster, reverse is rare | Not typical for image libraries |

---

## 2. Filter/Operation Comparison Matrix

### vs libvips (primary performance benchmark)

| Category | libvips | rasmcore | Gap |
|----------|---------|----------|-----|
| **Blur** | gaussblur, sharpen | blur, bokeh_blur, motion_blur, zoom_blur, gaussian_blur_cv | **rasmcore ahead** |
| **Sharpen** | sharpen | sharpen | Parity |
| **Edge detect** | canny, sobel, scharr, prewitt | canny, sobel, scharr, laplacian | prewitt missing (P2) |
| **Morphology** | morph (generic) | erode, dilate, open, close, gradient, tophat, blackhat | **rasmcore ahead** — 7 specific ops |
| **Histogram** | hist_equal, hist_local, hist_match, hist_find, hist_cum, hist_entropy, hist_norm, hist_plot | equalize, CLAHE, histogram, histogram_match, statistics | hist_entropy, hist_cum, hist_plot missing (P2) |
| **Color space** | 30+ conversions (Lab, XYZ, HSV, CMYK, scRGB, Oklab, Oklch, Yxy) | Lab, XYZ, HSV, HSL, YCbCr, CMYK, linear RGB, Adobe RGB, ProPhoto, Oklab | ~~Oklab done~~ Yxy, scRGB missing (P2) |
| **Resize** | resize, thumbnail, shrink, reduce, affine, similarity, mapim | resize (13 filters), crop, affine, auto-orient | mapim (generic remap) missing (P2) |
| **Composite** | composite, composite2 (many blend modes) | composite, blend (19 modes) | Parity (rasmcore has 19 vs libvips 11) |
| **Convolution** | conv, convi, convf, convsep, conva, compass | convolve (with separable detection) | Parity |
| **Noise/denoise** | — | median, bilateral, guided, NLM, film_grain, perlin, simplex | **rasmcore ahead** |
| **Tone mapping** | — | tonemap_reinhard, tonemap_drago, tonemap_filmic | **rasmcore ahead** |
| **Drawing** | draw_circle, draw_flood, draw_image, draw_line, draw_mask, draw_rect, draw_smudge | draw_line, draw_rect, draw_circle, draw_text, draw_text_ttf, flood_fill | draw_image, draw_mask, draw_smudge missing (P2) |
| **Smart operations** | smartcrop | smart_crop, seam_carve, inpaint, perspective_correct | **rasmcore ahead** |
| **Frequency domain** | fwfft, invfft, freqmult, phasecor | frequency_low, frequency_high (Gaussian separation) | **FFT missing** (P1) — no true Fourier transform |
| **Mosaicing** | mosaic, match, globalbalance | concat | mosaic/stitching missing (P2) |
| **Hough transform** | hough_line, hough_circle | hough_lines_p | hough_circle missing (P2) |
| **Statistics** | avg, deviate, max, min, percent, stats, profile, project, countlines | statistics, histogram | percent, profile, project, countlines missing (P2) |
| **Text rendering** | text (Pango) | draw_text (bitmap 8x16) | **Pango/FreeType text missing** (P1) — only bitmap font |
| **LUT/CLUT** | maplut, buildlut, hald-clut | color_lut (3D CLUT), apply_cube_lut, apply_hald_lut | ~~hald-clut done~~ maplut, buildlut missing (P2) |
| **Fill nearest** | fill_nearest | — | Missing (P2) |
| **SDF** | sdf | — | Missing (P2) — signed distance field |
| **Label regions** | labelregions | connected_components | Parity |
| **ICC profiles** | profile_load, icc_import, icc_export, icc_transform | ICC read (embedded in decode) | **ICC transform/apply missing** (P0) — can read but not apply |

### vs ImageMagick (feature-completeness benchmark)

| Category | ImageMagick | rasmcore | Gap |
|----------|-------------|----------|-----|
| **Blur variants** | adaptive-blur, bilateral-blur, blur, gaussian-blur, motion-blur, rotational-blur, selective-blur | blur, bokeh_blur, motion_blur, zoom_blur, bilateral, gaussian_blur_cv | adaptive-blur, rotational-blur, selective-blur missing (P2) |
| **Sharpen** | adaptive-sharpen, sharpen, unsharp | sharpen | adaptive-sharpen, unsharp mask as separate op (P2) |
| **Distortion** | distort (Affine, Perspective, Polar, Arc, Barrel, Shepards, BilinearForward/Reverse, Polynomial, Cylinder2Plane, Plane2Cylinder, DePolar) | barrel, spherize, polar, depolar, wave, ripple, swirl, perspective_warp, affine | Shepards, BilinearForward/Reverse, Polynomial, Cylinder2Plane missing (P2) |
| **Segmentation** | kmeans, segment, mean-shift, connected-components | connected_components, quantize | **k-means clustering missing** (P1), mean-shift missing (P2) |
| **Noise** | +noise (Gaussian, Impulse, Laplacian, Multiplicative, Poisson, Uniform), wavelet-denoise, despeckle | perlin, simplex, film_grain, NLM denoise, bilateral, median | +noise (add specific noise types) missing (P1), wavelet denoise missing (P2) |
| **Animation** | coalesce, deconstruct, morph (frame interp), delay, dispose, loop | decode_all_frames, encode_sequence, frame_count | **frame morph/interpolation missing** (P1), coalesce/deconstruct (P2) |
| **Compare** | compare -metric (PSNR, SSIM, RMSE, MAE, NCC, PHASH, FUZZ, AE, MSE, MEPP) | psnr, ssim, rmse, mae, delta_e_cie76 | NCC, PHASH, FUZZ, AE, MEPP missing (P2) |
| **FX expressions** | -fx (per-pixel math expressions) | — | **Missing** (P2) — pixel-level scripting |
| **White balance** | -white-balance | white_balance_gray_world, white_balance_temperature | Parity |
| **Color LUT** | -clut, -hald-clut | color_lut (3D CLUT), apply_cube_lut, apply_hald_lut | ~~hald-clut done~~ |
| **Dither** | ordered-dither (many patterns), random-threshold | dither_floyd_steinberg, dither_ordered | random-threshold missing (P2) |
| **Canvas/layout** | append, border, frame, smush, montage | concat_horizontal, concat_vertical, concat_grid, pad | montage/contact sheet missing (P2) |
| **Evaluate** | -evaluate (Add, Subtract, Multiply, Divide, And, Or, Xor, Min, Max, Pow, Log, Mean, Median, etc.) | — | **Per-pixel arithmetic missing** (P1) — useful for compositing workflows |

### vs sharp (web developer benchmark)

| Feature | sharp | rasmcore | Gap |
|---------|-------|----------|-----|
| **Format support** | JPEG, PNG, WebP, AVIF, GIF, TIFF, SVG, HEIF, RAW | 19 formats | **rasmcore ahead** (more formats) |
| **Resize** | resize with cover/contain/fill/inside/outside | resize with 13 filter options | Parity (different approach) |
| **Operations** | blur, sharpen, median, flatten, gamma, negate, normalize, clahe, convolve, threshold, boolean, linear, recomb, modulate, tint, greyscale, removeAlpha, ensureAlpha, extractChannel, joinChannel, bandbool, trim, erode, dilate | 128 registered filters | **rasmcore far ahead** |
| **Pipeline** | Sequential operation chain | Node graph with spatial cache | **rasmcore ahead** (more sophisticated) |
| **Animated** | GIF, WebP (encode+decode), APNG | GIF, WebP (decode), APNG, TIFF | WebP animated encode missing |
| **Streaming** | Stream I/O | Demand-driven tile pipeline | **rasmcore ahead** |

### vs Pillow (data science/ML benchmark)

| Feature | Pillow | rasmcore | Gap |
|---------|--------|----------|-----|
| **Formats** | ~30 decode, ~20 encode | 19 decode, 17 encode | Pillow has more exotic formats |
| **Filters** | BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EMBOSS, FIND_EDGES, SHARPEN, SMOOTH, plus ImageOps | 128 registered filters | **rasmcore far ahead** |
| **Transforms** | resize, rotate, crop, transpose, perspective, affine | 13+ transforms | Parity |
| **Enhancement** | Brightness, Color, Contrast, Sharpness (ImageEnhance) | brightness, contrast, saturate, sharpen, etc. | Parity |
| **Drawing** | ImageDraw (text, shapes, polygons, arcs, chords, pie slices) | draw_line, draw_rect, draw_circle, draw_text | **Polygon/arc/pie drawing missing** (P2) |
| **Quantize** | quantize (median cut, octree, fast octree, libimagequant) | quantize (median cut), Floyd-Steinberg, ordered | libimagequant missing (P2) |
| **Sequence** | GIF, APNG, TIFF sequences | GIF, APNG, TIFF sequences | Parity |
| **Font rendering** | FreeType/TrueType text | Bitmap 8x16 + TTF (draw_text_ttf) | Parity |

---

## 3. Pipeline/Architecture Comparison

| Feature | rasmcore | libvips | ImageMagick | sharp |
|---------|----------|---------|-------------|-------|
| **Architecture** | Node graph + spatial cache | Demand-driven pipeline | Pixel list (sequential) | Sequential chain (wraps libvips) |
| **Memory model** | Tile-based, lazy evaluation | Tile-based, demand-driven | Full image in memory | Streaming (via libvips) |
| **Multi-thread** | Host-driven parallelism (Level 2 API) | Automatic thread pool | OpenMP | libvips threads |
| **WASM support** | Native (primary target) | No | No | No |
| **GPU** | Planned (wgpu) | No | OpenCL (limited) | No |
| **Plugin system** | WIT component model | loadable modules | delegate system | — |
| **16-bit support** | Full (Gray16, RGB16, RGBA16) | Full | Full | Limited |
| **HDR pipeline** | EXR + tonemap operators | Full (scRGB, float) | Full (HDRI build) | Limited |

---

## 4. Prioritized Gap List

### P0 — Critical for Competitive Parity

| # | Gap | Impact | Scope | Notes |
|---|-----|--------|-------|-------|
| 1 | **AVIF decode** | Can encode but not decode — broken roundtrip | M | Need rav1d or dav1d integration |
| 2 | **ICC profile transform/apply** | Can read ICC but not color-manage images | M | Need lcms2 equivalent or moxcms expansion |
| ~~3~~ | ~~TrueType/OpenType font rendering~~ | ~~draw_text only has bitmap 8x16 font~~ | — | **DONE** — draw_text_ttf registered |

### P1 — Important for Feature Parity

| # | Gap | Impact | Scope | Notes |
|---|-----|--------|-------|-------|
| 3 | **JPEG XL encode** | Growing adoption, we only decode | M | jxl-oxide or libjxl binding |
| 4 | **FFT/frequency domain** | No true Fourier transform (only Gaussian separation) | M | Pure Rust FFT (rustfft crate) |
| 5 | **k-means color clustering** | IM has it, useful for palette/segmentation | S | Standard algorithm |
| 6 | **Add-noise operations** | Can't add Gaussian/salt-pepper/Poisson noise | S | Common test/augmentation need |
| 7 | **Animated WebP encode** | GIF replacement blocked | M | image-webp crate limitation — may need alternative |
| 8 | **Per-pixel arithmetic** | No evaluate/fx equivalent for compositing math | S | Add evaluate() with basic ops |
| 9 | **Frame interpolation/morphing** | No animated frame generation | M | Useful for animation creation |
| 10 | **RAW/DNG decode** | Photography workflow gap | L | Complex — dcraw or LibRaw equivalent |
| 11 | **Fix Gray8-output filter pipeline bug** | charcoal/sobel/canny/grayscale break downstream in pipeline | S | Need mapper registration or auto-convert |

### Closed since 2026-03-30
- ~~Oklab/Oklch~~ **done** — rgb_to_oklab/oklab_to_rgb in color_spaces.rs
- ~~Montage/contact sheet~~ **done** — concat_horizontal, concat_vertical, concat_grid
- ~~TrueType font~~ **done** — draw_text_ttf registered
- ~~Hald CLUT~~ **done** — apply_hald_lut registered filter
- ~~Delta E 2000~~ **done** — delta_e_2000 in color_spaces.rs
- ~~Drawing primitives~~ **done** — draw_line, draw_rect, draw_circle all registered
- ~~Filter registration~~ **done** — 128 registered (was 49)

### P2 — Nice to Have

| # | Gap | Impact | Scope | Notes |
|---|-----|--------|-------|-------|
| 12 | Adaptive blur/sharpen | Edge-aware variants | S | |
| 13 | Prewitt edge detector | Minor edge detection variant | S | |
| 14 | Hough circle detection | Only have line detection | S | |
| 15 | Wavelet denoise | Alternative denoise method | M | |
| 16 | draw_polygon, draw_arc, draw_pie | Pillow has these | S | |
| 17 | Histogram entropy/cumulative | libvips stats operations | S | |
| 18 | SDF (signed distance field) | Niche but libvips has it | S | |
| 19 | mapim (generic remap) | Arbitrary pixel remapping | M | |
| 20 | Image mosaicing/stitching | libvips feature | L | |
| 21 | PSD read | Photoshop format | M | |
| 22 | PDF read | Document imaging | M | |
| 23 | NCC/PHASH metrics | Additional comparison metrics | S | |
| 24 | libimagequant palette | Better quantization quality | S | |

---

## 5. Competitive Position Summary

### Strengths (rasmcore leads)
- **Filter count**: 128 registered filters — far more than sharp (~30), Pillow (~15), comparable to libvips (~300 but many are arithmetic/stats)
- **Color grading**: Cinematography-grade (ASC CDL, lift/gamma/gain, curves, split toning, 3 tonemappers, film grain)
- **Content-aware**: Seam carving, smart crop, inpainting, perspective correction, lens undistortion
- **Denoising**: 4 denoise algorithms (median, bilateral, guided, NLM)
- **Artistic effects**: Oil paint, charcoal, halftone, Kuwahara, pixelate, emboss, solarize + 6 distortion effects
- **WASM-native**: Only product in this comparison that targets WASM as primary
- **Architecture**: Spatial caching pipeline + optional layer cache with BLAKE3 content hashing
- **Format breadth**: QOI, FITS, DDS — formats others don't have
- **Plugin system**: Proc-macro registration with UI parameter metadata (ConfigParams) — unique
- **3D LUT**: .cube + Hald + LUT composition (chaining) — most complete LUT support of any library
- **19 blend modes**: More than libvips (11), competitive with sharp (20+)
- **Drawing**: Line, rect, circle, text (bitmap + TTF), flood fill — all registered

### Weaknesses (rasmcore trails)
- **AVIF decode**: Critical gap — can't roundtrip the most important modern web format
- **ICC color management**: Can read profiles but can't apply them — images may display wrong
- **FFT**: No frequency domain operations
- **Format count**: IM has 100+ formats (though most are obscure)
- **Pipeline Gray8 bug**: Filters that change output format (charcoal, sobel, grayscale) break downstream pipeline nodes

### Competitive Positioning
- **vs libvips**: Close on filters, behind on color management and FFT, ahead on artistic/content-aware/grading
- **vs ImageMagick**: Behind on format count and scripting (-fx), ahead on architecture, grading, and WASM
- **vs sharp**: Far ahead on filter count (128 vs ~30) and pipeline sophistication
- **vs Pillow**: Far ahead on filters, now at parity on font rendering
- **vs image-rs**: Far ahead on everything except ecosystem adoption (image-rs has 220M downloads)

---

## 6. Recommended Next Tracks (Priority Order — Updated 2026-03-31)

### Completed since last audit
- ~~TrueType font rendering~~ **done**
- ~~Oklab/Oklch color space~~ **done**
- ~~Montage/contact sheet~~ **done** (concat_horizontal/vertical/grid)
- ~~Hald CLUT~~ **done**
- ~~Delta E 2000~~ **done**
- ~~Drawing primitives~~ **done**
- ~~Filter registration (30 filters)~~ **done** (128 total now)

### Remaining priorities
1. **AVIF decode** (P0, M) — Complete the AVIF roundtrip
2. **ICC profile application** (P0, M) — Color-managed pipeline
3. **Fix Gray8 pipeline bug** (P1, S) — charcoal/sobel/canny/grayscale need mapper registration
4. **Add-noise operations** (P1, S) — Quick win, common need
5. **k-means clustering** (P1, S) — Quick win, segmentation
6. **FFT operations** (P1, M) — Frequency domain processing
7. **JPEG XL encode** (P1, M) — Complete JXL roundtrip
8. **Per-pixel arithmetic** (P1, S) — Compositing math
9. **Animated WebP encode** (P1, M) — GIF replacement
