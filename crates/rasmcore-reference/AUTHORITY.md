# Reference Tool Authority Catalog

This document maps every rasmcore filter to its authoritative reference tool
for cross-validation. The reference tool is the "ground truth" — if our
implementation disagrees with the reference, our code is assumed wrong unless
the discrepancy is documented and justified.

## Tool Overview

| Tool | Version | Linear f32 Mode | Best For |
|------|---------|-----------------|----------|
| **ImageMagick** | 7.1.1+ | `-colorspace Linear`, EXR I/O | Point ops, distortion, general purpose |
| **vips** | 8.15+ | `--linear` flag, native f32 | Color ops, spatial ops, high-performance batch |
| **OpenCV** | 4.8+ | Native float32 (CV_32FC4) | Edge detection, morphology, spatial filters, CLAHE |
| **GIMP** | 2.10+ | 32-bit float precision, Script-Fu batch | Fallback for ops not in other tools |
| **DaVinci Resolve** | 19+ | ACES/Linear working space | Color grading (CDL, curves, LGG, tone mapping) |
| **OpenColorIO** | 2.3+ | Native f32 pipeline | LUT application, color space transforms |
| **W3C Specifications** | CSS Compositing L1, CSS Filter Effects L1 | N/A (math spec) | Blend modes, hue-rotate, sepia matrix |
| **ISO/ITU Standards** | BT.709, BT.2020 | N/A (math spec) | Luma coefficients, transfer functions |

### Installation

```bash
# macOS
brew install imagemagick libvips opencv gimp

# Linux (Ubuntu/Debian)
sudo apt install imagemagick libvips-tools python3-opencv gimp
```

---

## Adjustment (14 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `brightness` | ImageMagick | `magick in.exr -colorspace Linear -evaluate Add {amount} out.exr` | Unambiguous additive op |
| `contrast` | ImageMagick | `magick in.exr -colorspace Linear -brightness-contrast 0x{amount*100}` | Linear ramp around 0.5 |
| `gamma` | ImageMagick | `magick in.exr -colorspace Linear -gamma {gamma}` | Standard power curve |
| `exposure` | DaVinci Resolve | Offset wheel in linear mode: `×2^ev` | Professional EV standard |
| `invert` | ImageMagick | `magick in.exr -colorspace Linear -negate out.exr` | Mathematically unambiguous |
| `levels` | ImageMagick | `magick in.exr -colorspace Linear -level {black}%,{white}%,{gamma} out.exr` | Matches PS Levels dialog |
| `posterize` | ImageMagick | `magick in.exr -colorspace Linear -posterize {levels} out.exr` | Standard quantization |
| `solarize` | ImageMagick | `magick in.exr -colorspace Linear -solarize {threshold*100}% out.exr` | IM invented this op |
| `sigmoidal_contrast` | ImageMagick | `magick in.exr -sigmoidal-contrast {strength},{midpoint*100}% out.exr` | IM canonical implementation |
| `dodge` | W3C + OpenCV | Linear dodge: `out = in / (1 - amount)` | Standard blend mode math |
| `burn` | W3C + OpenCV | Linear burn: `out = 1 - (1 - in) / amount` | Standard blend mode math |

### Research Notes — Exposure

DaVinci Resolve uses pure multiplicative exposure (`×2^ev`) in linear space, matching
our implementation exactly. Nuke's EXPTool does the same. ImageMagick has no native
EV operation — it would be `-evaluate Multiply {2^ev}`. Resolve is the authority
because exposure is fundamentally a color grading operation.

## Color (22 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `sepia` | W3C CSS Filter Effects | Matrix: R'=.393R+.769G+.189B (clamped) | Formally specified |
| `saturate` | Ottosson 2020 / CSS Color L4 | OKLCH chroma scaling — perceptually uniform | Bottosson OKLab, W3C standard |
| `saturate_hsl` | vips | `vips colourspace` HSL saturation model (legacy) | Native f32, HSL hexcone |
| `hue_rotate` | W3C CSS Filter Effects | YIQ rotation matrix (Section 2) | Formally specified |
| `vibrance` | DaVinci Resolve | Vibrance in Color page (saturation-weighted) | Professional grading tool |
| `channel_mixer` | DaVinci Resolve | Color Mixer (3×3 matrix) | Mathematically unambiguous |
| `white_balance_temperature` | vips | `vips colourspace` temperature shift | Native f32 pipeline |
| `white_balance_gray_world` | OpenCV | `cv2.xphoto.createGrayworldWB()` | Standard algorithm |
| `colorize` | GIMP | Colors > Colorize (HSL tint) | GIMP is canonical for this |
| `modulate` | ImageMagick | `magick -modulate {brightness},{saturation},{hue}` | IM canonical |
| `photo_filter` | DaVinci Resolve | Color warming/cooling filter | Professional color tool |
| `gradient_map` | GIMP | Colors > Gradient Map | GIMP canonical |
| `selective_color` | DaVinci Resolve | Qualifier + color correction | Professional grading |
| `replace_color` | GIMP | Select by Color + recolor | GIMP canonical |
| `quantize` | ImageMagick | `magick -colors {n} -dither None` | IM canonical |
| `kmeans_quantize` | OpenCV | `cv2.kmeans()` on pixel data | Standard k-means |
| `dither_floyd_steinberg` | ImageMagick | `magick -dither FloydSteinberg -colors {n}` | Standard algorithm |
| `dither_ordered` | ImageMagick | `magick -dither Riemersma -ordered-dither` | Standard Bayer matrix |
| `sparse_color` | ImageMagick | `magick -sparse-color` various methods | IM canonical |
| `lab_adjust` | OpenCV | Convert to Lab, adjust L/a/b, convert back | CIE Lab standard |
| `lab_sharpen` | OpenCV | Sharpen L channel in Lab space | Professional technique |
| `match_color` | OpenCV | `cv2.xphoto.applyChannelGains()` | Standard color transfer |
| `aces_idt` / `aces_odt` / `aces_cct_to_cg` / `aces_cg_to_cct` | OpenColorIO | ACES reference transforms | ACES standard |

### Research Notes — Saturation Model

**Decision: OKLCH perceptual model (default) + HSL legacy**

The default `saturate` filter now uses OKLCH chroma scaling, which is
perceptually uniform across hues. The legacy HSL model is preserved
as `saturate_hsl` for backward compatibility.

- **OKLCH (Ottosson 2020)**: Scales chroma in OKLab cylindrical space.
  Equal factor changes produce equal perceived saturation changes for
  all hues. Simple — 3x3 matrix + cbrt, no iterative solver.
  Reference: https://bottosson.github.io/posts/oklab/
- **CSS Color Level 4**: W3C adopted OKLab/OKLCH in Section 8.
- **CSS `saturate()`**: Still uses the old HSL model (CSS Filter Effects L1).
  Our `saturate_hsl` matches this for web compatibility testing.
- **DaVinci Resolve**: Uses a perceptual saturation model tied to ACES.
- **Nuke**: SaturationNode uses luma-blend: `out = luma + factor * (in - luma)`.

### Research Notes — White Balance

**Decision: Simplified Planckian shift for `white_balance_temperature`**

- **Our impl**: `R *= 1 + shift*0.1, B *= 1 - shift*0.1` (linear approx)
- **DaVinci Resolve**: Full CIE D-illuminant calculation with CAT02 chromatic
  adaptation transform. Temperature in Kelvin, not arbitrary units.
- **Lightroom/ACR**: Similar to Resolve but with Adobe's proprietary adaptation.
- **vips**: `vips colourtemp` uses CIE daylight model.

Our simplified model is NOT professional-grade. The authority for validation is
vips (closest to our model), but AUTHORITY.md notes this as a known gap.
A future track should implement proper CIE CAT02/CAT16 adaptation matching Resolve.

## Spatial (18 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `gaussian_blur` | Nuke/Resolve | 5*sigma truncation (99.99994% energy), f32 separable convolution | Nuke ~4.4*sigma, Resolve 4*sigma; we use 5*sigma for f32-exact quality |
| `box_blur` | OpenCV | `cv2.blur(img, (k,k))` on float32 | Uniform averaging kernel |
| `median` | OpenCV | `cv2.medianBlur(img, k)` | Standard median filter |
| `bilateral` | OpenCV | `cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)` | Tomasi-Manduchi formulation |
| `sharpen` | vips | `vips sharpen` (unsharp mask model) | Standard USM |
| `high_pass` | OpenCV | `img - cv2.GaussianBlur(img)` | Standard high-pass via subtraction |
| `motion_blur` | OpenCV | Directional kernel convolution | Standard technique |
| `lens_blur` | OpenCV | Disc kernel convolution | Standard technique |
| `bokeh_blur` | OpenCV | Shaped kernel (hexagonal aperture) | Standard technique |
| `spin_blur` | ImageMagick | `magick -radial-blur {angle}` | IM canonical |
| `zoom_blur` | ImageMagick | Custom radial sampling | Technique standard |
| `smart_sharpen` | vips | Deconvolution-based sharpening | Professional technique |
| `displacement_map` | ImageMagick | `magick -displace` | IM canonical |
| `convolve` | OpenCV | `cv2.filter2D(img, -1, kernel)` | Standard 2D convolution |
| `nlm_denoise` | OpenCV | `cv2.fastNlMeansDenoisingColored()` on float32 | Buades et al. algorithm |
| `tilt_shift` | Composite | Selective blur + gradient mask | Technique, not single-tool |
| `frequency_low` / `frequency_high` | OpenCV | Gaussian blur / subtract from original | Standard frequency decomposition |

### Research Notes — Bilateral Filter

**Decision: Tomasi-Manduchi (1998) formulation, matching OpenCV**

OpenCV implements the original bilateral filter from Tomasi & Manduchi (1998).
Parameters: kernel size `d`, color sigma `sigmaColor`, spatial sigma `sigmaSpace`.
DaVinci Resolve's spatial NR uses a different formulation (wavelet-based).
We align with OpenCV as it's the standard textbook implementation and our
filter uses the same formulation.

### Research Notes — Gaussian Blur Kernel

**Decision: Truncation at 3*sigma, matching OpenCV default**

OpenCV's `getGaussianKernel` truncates at `ksize = round(sigma * 6 + 1) | 1`.
This is effectively 3*sigma on each side. Our radius parameter maps to sigma,
and the kernel is truncated the same way. This is standard practice.

## Enhancement (16 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `auto_level` | ImageMagick | `magick -colorspace Linear -auto-level` | Standard min/max stretch |
| `equalize` | OpenCV | `cv2.equalizeHist()` (per channel on float32) | Standard CDF remap |
| `normalize` | ImageMagick | `magick -normalize` (with percentile clipping) | Standard percentile stretch |
| `clahe` | OpenCV | `cv2.createCLAHE(clipLimit, tileGridSize)` | Pizer et al. algorithm |
| `shadow_highlight` | DaVinci Resolve | Shadow/Highlight in Color page | Professional grading |
| `clarity` | DaVinci Resolve | Clarity/Mid Detail | Professional grading |
| `dehaze` | OpenCV | Dark channel prior (He et al. 2009) | Standard algorithm |
| `vignette` / `vignette_powerlaw` | ImageMagick | `magick -vignette` | Standard radial falloff |
| `retinex_ssr` / `retinex_msr` / `retinex_msrcr` | OpenCV | `cv2.xphoto` retinex or manual impl | Land's Retinex theory |
| `pyramid_detail_remap` | vips | Laplacian pyramid decomposition | Standard multi-scale |

### Research Notes — CLAHE

**Decision: OpenCV convention (Pizer et al. 1987)**

OpenCV's CLAHE uses `clipLimit=2.0, tileGridSize=(8,8)` as defaults.
Our implementation matches this convention. The clip limit is in histogram
bin counts (proportional to tile size), not a normalized value.

## Edge Detection (8 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `sobel` | OpenCV | `cv2.Sobel(img, cv2.CV_32F, dx, dy)` | Standard 3×3 Sobel kernel |
| `scharr` | OpenCV | `cv2.Scharr(img, cv2.CV_32F, dx, dy)` | Improved Sobel coefficients |
| `canny` | OpenCV | `cv2.Canny()` (on 8-bit internally) | Canny 1986 algorithm |
| `laplacian` | OpenCV | `cv2.Laplacian(img, cv2.CV_32F)` | Standard LoG kernel |
| `threshold_binary` | OpenCV | `cv2.threshold(img, thresh, 1.0, cv2.THRESH_BINARY)` | Standard thresholding |
| `otsu_threshold` | OpenCV | `cv2.threshold(img, 0, 1.0, cv2.THRESH_OTSU)` | Otsu 1979 algorithm |
| `triangle_threshold` | OpenCV | `cv2.threshold(img, 0, 1.0, cv2.THRESH_TRIANGLE)` | Zack et al. 1977 |
| `adaptive_threshold` | OpenCV | `cv2.adaptiveThreshold()` | Standard block-mean method |

## Morphology (8 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `dilate` | OpenCV | `cv2.dilate(img, kernel)` | Standard morphological dilation |
| `erode` | OpenCV | `cv2.erode(img, kernel)` | Standard morphological erosion |
| `morph_open` | OpenCV | `cv2.morphologyEx(MORPH_OPEN)` | Erosion then dilation |
| `morph_close` | OpenCV | `cv2.morphologyEx(MORPH_CLOSE)` | Dilation then erosion |
| `morph_gradient` | OpenCV | `cv2.morphologyEx(MORPH_GRADIENT)` | Dilation minus erosion |
| `morph_tophat` | OpenCV | `cv2.morphologyEx(MORPH_TOPHAT)` | Image minus opening |
| `morph_blackhat` | OpenCV | `cv2.morphologyEx(MORPH_BLACKHAT)` | Closing minus image |
| `skeletonize` | OpenCV | Iterative thinning (Zhang-Suen) | Standard algorithm |

## Grading (4+ filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `tonemap_reinhard` | OpenCV | `cv2.createTonemapReinhard()` | Reinhard et al. 2002 |
| `tonemap_drago` | OpenCV | `cv2.createTonemapDrago()` | Drago et al. 2003 |
| `tonemap_filmic` | DaVinci Resolve | ACES Filmic tone curve | ACES reference |
| `film_grain_grading` | DaVinci Resolve | Film grain in Resolve | Professional tool |

Note: Curves, CDL, LGG are manually registered grading filters. They follow
ASC CDL (SMPTE standard) and standard cubic spline interpolation respectively.

| Filter | Authority | Spec | Rationale |
|--------|-----------|------|-----------|
| Curves (master, per-channel) | DaVinci Resolve | Cubic Hermite spline interpolation | Professional standard |
| ASC CDL | SMPTE | ASC CDL v1.2 specification | Industry standard |
| Lift/Gamma/Gain | DaVinci Resolve | Primary wheels in Color page | Professional standard |

## Composite (4 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `premultiply` | W3C | Alpha premultiplication: `R' = R * A` | Mathematically unambiguous |
| `unpremultiply` | W3C | Reverse: `R' = R / A` (safe div) | Mathematically unambiguous |
| `blend` | W3C CSS Compositing L1 | All 30+ blend modes formally specified | W3C canonical for web; PDF spec (ISO 32000) for print |
| `blend_if` | DaVinci Resolve | Luminance-based opacity masking | Professional tool |

### Research Notes — Blend Modes

**Decision: W3C CSS Compositing and Blending Level 1 (2015) + ISO 32000-2 (PDF 2.0)**

Both specs define the same formulas for standard blend modes (multiply, screen,
overlay, etc.). DaVinci Resolve and Nuke implement these same formulas.
For modes beyond the CSS spec (linear light, pin light, vivid light), we use
the Adobe/PDF spec (ISO 32000-2) which is the superset.

## Effect (16 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `emboss` | ImageMagick | `magick -emboss {radius}` | Standard convolution kernel |
| `pixelate` | ImageMagick | `magick -scale {1/n}x -scale {n}x` | Block averaging |
| `gaussian_noise` | OpenCV | `cv2.randn()` on float32 | Standard Gaussian RNG |
| `uniform_noise` | OpenCV | `cv2.randu()` on float32 | Standard uniform RNG |
| `poisson_noise` | OpenCV | Poisson sampling | Standard distribution |
| `salt_pepper_noise` | OpenCV | Bernoulli impulse noise | Standard definition |
| `film_grain` | DaVinci Resolve | Film grain emulation | Professional tool |
| `glitch` | N/A | Custom artistic effect | No standard reference |
| `light_leak` | N/A | Custom artistic effect | No standard reference |
| `halftone` | GIMP | Filters > Distorts > Newsprint | Standard halftone |
| `oil_paint` | OpenCV | `cv2.xphoto.oilPainting()` | Standard Kuwahara variant |
| `charcoal` | ImageMagick | `magick -charcoal {radius}` | IM canonical |
| `sponge` | GIMP | Artistic filter | GIMP reference |
| `chromatic_aberration` | N/A | Geometric channel offset | Standard lens model |
| `chromatic_split` | N/A | Artistic channel separation | No standard |
| `mirror_kaleidoscope` | ImageMagick | Mirror + rotation | Geometric transform |

## Distortion (10 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `barrel` | OpenCV | `cv2.undistort()` Brown-Conrady model | Standard lens distortion |
| `swirl` | ImageMagick | `magick -swirl {degrees}` | IM canonical |
| `wave` | ImageMagick | `magick -wave {amplitude}x{wavelength}` | IM canonical |
| `ripple` | ImageMagick | Sinusoidal displacement | Standard formula |
| `polar` / `depolar` | OpenCV | `cv2.linearPolar()` / `cv2.logPolar()` | Standard coordinate transform |
| `spherize` | GIMP | Filters > Distorts > Spherize | GIMP canonical |
| `liquify` | DaVinci Resolve | Forward warp mesh | Professional tool |
| `mesh_warp` | DaVinci Resolve | Free-form mesh deformation | Professional tool |

## Evaluate (9 filters)

| Filter | Authority | Spec | Rationale |
|--------|-----------|------|-----------|
| `evaluate_add` | N/A | `out = in + constant` | Mathematically unambiguous — no external reference needed |
| `evaluate_subtract` | N/A | `out = in - constant` | Mathematically unambiguous |
| `evaluate_multiply` | N/A | `out = in * constant` | Mathematically unambiguous |
| `evaluate_divide` | N/A | `out = in / constant` | Mathematically unambiguous |
| `evaluate_pow` | N/A | `out = in ^ constant` | Mathematically unambiguous |
| `evaluate_abs` | N/A | `out = abs(in)` | Mathematically unambiguous |
| `evaluate_min` | N/A | `out = min(in, constant)` | Mathematically unambiguous |
| `evaluate_max` | N/A | `out = max(in, constant)` | Mathematically unambiguous |
| `evaluate_log` | N/A | `out = log(in)` | Mathematically unambiguous |

## Alpha (3 filters)

| Filter | Authority | Spec | Rationale |
|--------|-----------|------|-----------|
| `add_alpha` | N/A | `alpha = constant` | Trivial — no reference needed |
| `remove_alpha` | N/A | `alpha = 1.0` | Trivial |
| `flatten` | W3C | Composite over solid background | Standard alpha compositing |

## Mask (7 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `luminance_range` | OpenCV | Threshold on BT.709 luminance | Standard luma computation |
| `color_range` | OpenCV | `cv2.inRange()` on HSV/RGB | Standard range masking |
| `gradient_mask` | N/A | Linear/radial gradient generation | Procedural — self-validating |
| `mask_apply` | W3C | Alpha premultiplication | Standard compositing |
| `mask_combine` | N/A | Min/max/multiply of mask channels | Mathematically unambiguous |
| `masked_blend` | W3C | Mask-weighted alpha compositing | Standard compositing |
| `feather` | OpenCV | Gaussian blur on alpha channel | Standard technique |

## Draw (7 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `draw_line` | OpenCV | `cv2.line()` (Bresenham's algorithm) | Standard rasterization |
| `draw_circle` | OpenCV | `cv2.circle()` (midpoint algorithm) | Standard rasterization |
| `draw_rect` | OpenCV | `cv2.rectangle()` | Standard rasterization |
| `draw_ellipse` | OpenCV | `cv2.ellipse()` | Standard rasterization |
| `draw_arc` | OpenCV | `cv2.ellipse()` with angle range | Standard rasterization |
| `draw_polygon` | OpenCV | `cv2.fillPoly()` | Standard scan-line fill |
| `solid_fill` | N/A | Constant color fill | Trivial |

## Generator (10 filters)

| Filter | Authority | Spec | Rationale |
|--------|-----------|------|-----------|
| All generators | N/A | Pure procedural math | Self-validating — no external reference. Output is deterministic from parameters. Visual inspection only. |

Generators: `checkerboard`, `gradient_linear`, `gradient_radial`, `perlin_noise`,
`simplex_noise`, `plasma`, `solid_color`, `fractal_noise`, `cloud_noise`, `pattern_fill`

## Scope (4 filters)

| Filter | Authority | Spec | Rationale |
|--------|-----------|------|-----------|
| All scopes | DaVinci Resolve | Scopes page (Waveform, Vectorscope, Parade, Histogram) | Professional visualization standard. Validated by visual comparison, not pixel-exact matching. |

Scopes: `histogram`, `waveform`, `vectorscope`, `parade`

## Analysis / Transform (5 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `smart_crop` | N/A | Content-aware energy minimization | Algorithm-specific, no standard |
| `seam_carve_width` / `seam_carve_height` | vips | `vips smartcrop` (seam carving mode) | Standard Avidan-Shamir algorithm |
| `perspective_correct` / `perspective_warp` | OpenCV | `cv2.getPerspectiveTransform()` + `cv2.warpPerspective()` | Standard projective transform |
| `connected_components` | OpenCV | `cv2.connectedComponents()` | Standard labeling algorithm |
| `harris_corners` | OpenCV | `cv2.cornerHarris()` | Harris & Stephens 1988 |
| `hough_lines` | OpenCV | `cv2.HoughLinesP()` | Standard Hough transform |
| `template_match` | OpenCV | `cv2.matchTemplate()` | Standard template matching |
| `ca_remove` | OpenCV | Lateral chromatic aberration correction | Standard lens correction |

## Tool (7 filters)

| Filter | Authority | Command / Spec | Rationale |
|--------|-----------|---------------|-----------|
| `flood_fill` | OpenCV | `cv2.floodFill()` | Standard flood fill algorithm |
| `clone_stamp` | GIMP | Clone tool | GIMP canonical |
| `healing_brush` | GIMP | Healing tool (Poisson blending) | GIMP canonical |
| `red_eye_remove` | OpenCV | `cv2.xphoto` red-eye removal | Standard detection + correction |
| `smudge` | GIMP | Smudge tool | GIMP canonical |
| `sponge` | GIMP | Sponge tool (saturate/desaturate) | GIMP canonical |
| `liquify` | DaVinci Resolve | Face refinement / warp | Professional tool |

---

## Summary Statistics

| Category | Total | Has Authority | Self-Validating | No Standard |
|----------|-------|---------------|-----------------|-------------|
| Adjustment | 14 | 14 | 0 | 0 |
| Color | 22 | 22 | 0 | 0 |
| Spatial | 18 | 18 | 0 | 0 |
| Enhancement | 16 | 16 | 0 | 0 |
| Edge | 8 | 8 | 0 | 0 |
| Morphology | 8 | 8 | 0 | 0 |
| Grading | 7 | 7 | 0 | 0 |
| Composite | 4 | 4 | 0 | 0 |
| Effect | 16 | 12 | 0 | 4 |
| Distortion | 10 | 10 | 0 | 0 |
| Evaluate | 9 | 0 | 9 | 0 |
| Alpha | 3 | 1 | 2 | 0 |
| Mask | 7 | 4 | 3 | 0 |
| Draw | 7 | 6 | 1 | 0 |
| Generator | 10 | 0 | 10 | 0 |
| Scope | 4 | 4 | 0 | 0 |
| Analysis | 8 | 7 | 1 | 0 |
| Tool | 7 | 7 | 0 | 0 |
| **Total** | **178** | **148** | **26** | **4** |

- **148 filters** have an identified external authority tool
- **26 filters** are self-validating (pure math, trivial, or procedural)
- **4 filters** have no standard reference (artistic/custom effects: glitch, light_leak, chromatic_split, chromatic_aberration — these are validated by visual inspection only)

## Known Gaps and Future Work

1. **White balance**: Our `white_balance_temperature` uses a simplified linear
   approximation. Professional tools (Resolve, Lightroom) use CIE chromatic
   adaptation transforms (CAT02/CAT16). A future track should implement proper
   CIE adaptation as a separate filter or upgrade to the existing one.

2. **Saturation**: Our HSL model matches CSS/GIMP but not Resolve's perceptual
   model. Consider adding an "ACES saturation" filter for professional grading.

3. **Film grain**: Our implementation is procedural noise-based. Resolve uses
   scanned film stock profiles. Consider adding grain profiles as a future enhancement.

4. **Scopes**: Visual comparison only — pixel-exact matching is not meaningful
   for visualization outputs.
