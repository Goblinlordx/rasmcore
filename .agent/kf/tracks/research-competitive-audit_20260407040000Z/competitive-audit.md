# Competitive Feature Audit — rasmcore vs Professional Image Tools

**Date:** 2026-04-07
**Scope:** Adobe Photoshop, Adobe Lightroom, DaVinci Resolve, The Foundry Nuke, Affinity Photo, GIMP, libvips, ImageMagick

## rasmcore Current State

- **164 registered filters** across 15+ categories
- **14 codecs** (read/write: PNG, JPEG, WebP, BMP, TIFF, EXR, HDR, GIF, QOI, TGA, PNM, DDS, FITS, ICO)
- **GPU-first** pipeline with WGSL compute shaders + SIMD CPU fallback
- **f32 everywhere** — no 8-bit intermediate quantization
- **ACES color management** — ACEScg/ACEScct working spaces, RRT+ODT output transforms
- **WASM Component Model** — runs in browser, edge, cloud, native
- **Layer cache** — content-addressed, cross-pipeline
- **Analysis buffer protocol** — zero-readback GPU analysis chains
- **Brush engine** — pressure-sensitive, accumulation buffer, undo stack
- **ML integration** — model pack pattern, host-side inference
- **Typed resource refs** — fonts, LUTs, node refs through unified param system

---

## Feature Matrix

Legend: Y = implemented, P = partial, N = missing, - = not applicable

### Adjustments / Correction

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Brightness/Contrast | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Exposure (EV) | Y | Y | Y | Y | Y | Y | N | N | N |
| Levels (input/output) | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Curves | N | Y | Y | Y | Y | Y | Y | N | N |
| White Balance (temp/tint) | Y | Y | Y | Y | Y | Y | N | N | N |
| HSL/HSV per-channel | P | Y | Y | Y | Y | Y | Y | N | N |
| Vibrance | Y | Y | Y | Y | Y | Y | N | N | N |
| Shadow/Highlight recovery | Y | Y | Y | Y | Y | Y | N | N | N |
| Clarity/Local Contrast | Y | Y | Y | Y | N | Y | N | N | N |
| Dehaze | Y | Y | Y | N | N | Y | N | N | N |
| Tone mapping (Reinhard/Drago/Filmic) | Y | P | P | Y | Y | P | N | N | N |
| Photo filter / color overlay | Y | Y | N | Y | Y | Y | N | N | N |
| Selective color | Y | Y | N | Y | Y | Y | N | N | N |
| Channel mixer | N | Y | Y | Y | Y | Y | Y | N | Y |
| Gradient map | N | Y | N | Y | N | Y | Y | N | N |
| Color lookup (3D LUT) | Y | Y | Y | Y | Y | Y | N | N | N |
| ICC profile handling | N | Y | Y | Y | Y | Y | Y | Y | Y |

### Color Grading

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| CDL (Slope/Offset/Power) | Y | N | N | Y | Y | N | N | N | N |
| Lift/Gamma/Gain (color wheels) | N | N | N | Y | Y | N | N | N | N |
| Log grading (ACEScct) | Y | N | N | Y | Y | N | N | N | N |
| .cube LUT import | Y | Y | Y | Y | Y | Y | N | N | N |
| .clf (Common LUT Format) | Y | N | N | N | Y | N | N | N | N |
| Hald CLUT import | Y | N | N | N | N | N | Y | N | N |
| ACES pipeline (IDT/RRT/ODT) | Y | N | N | Y | Y | N | N | N | N |
| OCIO integration | N | N | N | Y | Y | N | N | N | N |
| Color space auto-conversion | Y | Y | Y | Y | Y | Y | N | N | N |
| Printer lights | N | N | N | Y | Y | N | N | N | N |
| Color warper | N | N | N | Y | N | N | N | N | N |

### Spatial Filters

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Gaussian blur | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Box blur | Y | Y | N | Y | Y | Y | Y | Y | Y |
| Motion blur | Y | Y | N | Y | Y | Y | Y | N | Y |
| Lens/bokeh blur | Y | Y | Y | Y | Y | Y | N | N | N |
| Spin/zoom blur | Y | Y | N | N | N | Y | N | N | Y |
| Tilt shift | Y | Y | Y | N | N | Y | N | N | N |
| Unsharp mask / Sharpen | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Smart sharpen | Y | Y | N | N | Y | Y | N | N | N |
| Lab sharpen | Y | N | N | N | N | N | N | N | N |
| High-pass filter | Y | Y | N | N | Y | Y | N | N | N |
| NLM denoise | Y | Y | Y | Y | Y | Y | Y | N | N |
| Bilateral denoise | Y | N | N | Y | Y | N | N | N | N |
| Median filter | Y | Y | N | Y | Y | Y | Y | N | Y |
| Frequency separation | Y | Y | N | N | N | Y | N | N | N |
| Retinex (SSR/MSR/MSRCR) | Y | N | N | N | N | N | N | N | N |
| CLAHE | Y | N | N | N | N | N | N | N | N |
| Pyramid detail remap | Y | N | N | N | N | N | N | N | N |

### Edge Detection / Morphology

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Sobel / Scharr / Laplacian | Y | Y | N | N | Y | N | Y | N | Y |
| Canny edge | Y | N | N | N | Y | N | N | N | N |
| Dilate / Erode | Y | Y | N | N | Y | Y | Y | N | Y |
| Open / Close | Y | Y | N | N | Y | N | Y | N | Y |
| Morphological gradient | Y | N | N | N | Y | N | N | N | N |
| Skeletonize (Zhang-Suen) | Y | N | N | N | N | N | N | N | N |
| Otsu / Triangle threshold | Y | N | N | N | N | N | Y | N | Y |
| Hough lines | Y | N | N | N | N | N | N | N | N |
| Connected components | Y | N | N | N | N | N | N | N | N |
| Template matching | Y | N | N | N | N | N | N | N | N |

### Transform / Geometry

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Crop | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Resize (bilinear/bicubic/lanczos) | N | Y | Y | Y | Y | Y | Y | Y | Y |
| Rotation (arbitrary angle) | N | Y | Y | Y | Y | Y | Y | Y | Y |
| Perspective warp | Y | Y | Y | N | Y | Y | Y | N | N |
| Barrel/pincushion distortion | Y | Y | Y | N | Y | Y | Y | N | Y |
| Spherize / Swirl / Ripple / Wave | Y | Y | N | N | Y | Y | Y | N | Y |
| Polar / Depolar | Y | Y | N | N | Y | N | Y | N | N |
| Mesh warp | Y | Y | N | N | Y | Y | N | N | N |
| Liquify | Y | Y | N | N | Y | Y | Y | N | N |
| Content-aware scale (seam carve) | Y | Y | N | N | N | N | Y | N | N |
| Lens correction profiles | N | Y | Y | Y | Y | Y | N | N | N |
| Flip / Mirror | N | Y | Y | Y | Y | Y | Y | Y | Y |

### Compositing / Blending

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Blend modes (25+) | Y | Y | N | Y | Y | Y | Y | Y | Y |
| Alpha compositing | Y | Y | N | Y | Y | Y | Y | Y | Y |
| Masking (luma, color range) | Y | Y | Y | Y | Y | Y | Y | N | N |
| Feather mask | Y | Y | Y | Y | Y | Y | Y | N | N |
| Gradient mask | Y | Y | Y | Y | Y | Y | Y | N | N |
| Blend If (luminosity masking) | Y | Y | N | Y | N | Y | N | N | N |
| Multi-layer compositing | P | Y | N | Y | Y | Y | Y | Y | Y |
| Keying (chroma key) | N | N | N | Y | Y | N | N | N | N |
| Merge/flatten layers | Y | Y | N | Y | Y | Y | Y | Y | Y |
| Displacement map | Y | Y | N | N | Y | Y | Y | N | N |

### Drawing / Paint

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Brush engine (pressure/tilt) | Y | Y | N | N | Y | Y | Y | N | N |
| Clone stamp | Y | Y | N | N | Y | Y | Y | N | N |
| Healing brush | Y | Y | Y | N | Y | Y | Y | N | N |
| Shape primitives (line/rect/circle) | Y | Y | N | N | N | Y | Y | N | Y |
| Text rendering (TTF) | Y | Y | Y | N | Y | Y | Y | N | Y |
| Pattern fill | Y | Y | N | N | Y | Y | Y | N | N |
| Gradient fill | Y | Y | Y | Y | Y | Y | Y | N | N |
| Flood fill | Y | Y | N | N | N | Y | Y | N | Y |
| Smudge/sponge | Y | Y | N | N | N | Y | Y | N | N |
| Stroke undo (tile-based) | Y | Y | N | N | Y | Y | Y | N | N |
| Brush presets | Y | Y | N | N | Y | Y | Y | N | N |

### Stylization / Effects

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Emboss | Y | Y | N | N | N | Y | Y | N | Y |
| Oil paint | Y | Y | N | N | N | Y | Y | N | N |
| Pixelate | Y | Y | N | N | N | Y | Y | N | Y |
| Halftone | Y | Y | N | N | N | Y | N | N | N |
| Charcoal | Y | Y | N | N | N | N | N | N | Y |
| Film grain | Y | Y | Y | Y | Y | Y | N | N | N |
| Glitch | Y | N | N | N | N | N | N | N | N |
| Solarize | Y | Y | N | N | N | Y | N | N | Y |
| Posterize | Y | Y | N | N | N | Y | Y | N | Y |
| Vignette | Y | Y | Y | Y | N | Y | Y | N | Y |
| Light leak | Y | N | N | N | N | N | N | N | N |

### Analysis / Scopes

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| Histogram | Y | Y | Y | Y | Y | Y | Y | N | Y |
| Waveform | Y | N | N | Y | N | N | N | N | N |
| Vectorscope | Y | N | N | Y | N | N | N | N | N |
| Parade (RGB) | Y | N | N | Y | N | N | N | N | N |
| Harris corners | Y | N | N | N | Y | N | N | N | N |
| Smart crop (energy-based) | Y | Y | N | N | N | N | N | Y | N |

### I/O / Codecs

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| PNG | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| JPEG | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| WebP | Y | Y | N | N | N | Y | Y | Y | Y |
| TIFF | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| EXR (OpenEXR) | Y | Y | N | Y | Y | Y | Y | Y | N |
| HDR (Radiance) | Y | Y | N | Y | Y | N | N | Y | Y |
| GIF | Y | Y | N | N | N | Y | Y | Y | Y |
| BMP | Y | Y | N | N | N | Y | Y | N | Y |
| AVIF | N | Y | N | N | N | Y | Y | Y | Y |
| HEIC/HEIF | N | Y | N | N | N | Y | Y | Y | Y |
| JPEG XL | N | P | N | N | N | N | Y | Y | Y |
| RAW (CR2/NEF/ARW) | N | Y | Y | Y | N | Y | Y | N | Y |
| PSD (layer support) | N | Y | - | N | N | Y | Y | N | N |
| DNG | N | Y | Y | Y | N | Y | Y | N | N |
| SVG | N | Y | N | N | N | Y | N | Y | Y |

### Architecture / Platform

| Feature | rasmcore | PS | LR | DR | Nuke | Affinity | GIMP | vips | IM |
|---------|----------|----|----|----|----- |----------|------|------|----|
| GPU compute pipeline | Y | Y | Y | Y | Y | Y | N | N | N |
| WASM portable | Y | N | N | N | N | N | N | N | N |
| f32 precision | Y | Y | P | Y | Y | Y | P | Y | P |
| ACES color management | Y | N | N | Y | Y | N | N | N | N |
| Content-addressed cache | Y | Y | Y | P | Y | N | N | N | N |
| Tiled processing | P | Y | Y | Y | Y | N | N | Y | N |
| Non-destructive pipeline | Y | Y | Y | Y | Y | Y | N | N | N |
| ML/AI integration | Y | Y | Y | Y | N | Y | N | N | N |
| Scripting/automation | N | Y | Y | Y | Y | Y | Y | N | N |
| Plugin architecture | N | Y | N | Y | Y | Y | Y | N | N |
| Batch processing | N | Y | Y | Y | Y | Y | Y | Y | Y |

---

## P0 Gaps (Critical for professional adoption)

1. **Curves adjustment** — The single most-used adjustment in every professional tool. Every competitor has it. rasmcore has levels and individual adjustments but no freeform curve editor.

2. **Resize with resampling** — Basic geometric transform missing. Bilinear, bicubic, Lanczos resampling not implemented. Every tool has this.

3. **Rotation** — Arbitrary angle rotation not implemented. Fundamental transform.

4. **Flip/Mirror** — Trivial but missing. Every tool has it.

5. **AVIF/HEIC codec** — Modern image formats. Browser and platform support is ubiquitous. AVIF especially critical for web deployment.

6. **RAW decode** — Professional photography workflow requires RAW support. Lightroom's primary use case.

7. **ICC profile handling** — Color-managed workflows require ICC profiles. All professional tools support this.

8. **Channel mixer** — Standard color adjustment present in every pro tool.

## P1 Gaps (Important for competitive parity)

9. **Lift/Gamma/Gain color wheels** — DaVinci Resolve's primary grading interface. Essential for colorists.

10. **OCIO integration** — OpenColorIO is the VFX industry standard. Nuke and Resolve both use it.

11. **Lens correction profiles** — Lightroom's lens correction is a key selling point. Adobe Lens Profile Creator data.

12. **Gradient map** — Popular adjustment layer in Photoshop.

13. **Keying (chroma key)** — Essential for compositing workflows (Nuke, Resolve).

14. **Scripting/automation** — Batch operations, macros, custom pipelines. PS has Actions, DR has Fusion scripting, Nuke has Python.

15. **Plugin architecture** — Extensibility for third-party developers.

16. **Batch processing** — Process multiple images with same pipeline.

17. **JPEG XL codec** — Next-gen codec gaining adoption.

18. **PSD file support** — Interoperability with Photoshop workflows.

19. **DNG / RAW processing** — Beyond just decode: demosaic, noise reduction, lens correction, tone curve.

## P2 Gaps (Differentiation / nice-to-have)

20. **Color warper** — DaVinci Resolve exclusive. Vector-based color manipulation.

21. **Printer lights** — Film-style grading tool (Resolve, Nuke).

22. **SVG import/export** — Vector graphics interop.

23. **Focus peaking / Zebra** — Real-time exposure and focus analysis overlays.

24. **History branching** — Non-linear undo with branching (PS history).

25. **Content-aware fill** — PS-exclusive intelligent fill.

---

## Architectural Assessment

| Gap | Type | Effort | Infrastructure needed |
|-----|------|--------|----------------------|
| Curves | Filter | Medium | Spline interpolation + GPU LUT shader |
| Resize | Transform | Medium | Resampling kernel infrastructure |
| Rotation | Transform | Medium | Affine transform + resampling |
| Flip/Mirror | Transform | Small | Trivial pixel reorder |
| AVIF/HEIC | Codec | Large | Native encoder/decoder or host delegation |
| RAW decode | Codec | Large | Demosaic pipeline, color matrix, lens correction |
| ICC profiles | Infrastructure | Large | Profile parsing, PCS conversion, rendering intent |
| Channel mixer | Filter | Small | 3x3 matrix multiply (already have infra) |
| LGG wheels | Filter | Small | Per-channel offset/power/gain (CDL variant) |
| OCIO | Infrastructure | Large | OpenColorIO integration or reimplementation |
| Lens correction | Filter + data | Large | Profile database + distortion model |
| Scripting | Infrastructure | Large | Rhai or similar embedded scripting |
| Plugin arch | Infrastructure | Large | Dynamic filter loading |
| Batch | Feature | Medium | Pipeline template + file iterator |

---

## Recommendations for Next Phase

### Immediate (close P0 gaps):
1. **Curves adjustment** — highest-impact single filter addition
2. **Resize + Rotation + Flip** — basic transform completeness
3. **Channel mixer** — trivial to add, high visibility
4. **AVIF codec** — via host delegation (browser has it built-in)

### Near-term (P1 competitive parity):
5. **LGG color wheels** — small effort, big grading impact
6. **Batch processing** — pipeline template + file iteration
7. **Scripting** — Rhai integration for pipeline scripting
8. **ICC profile support** — color-managed workflow completion

### Strategic (differentiation):
9. **RAW processing pipeline** — full photographer workflow
10. **OCIO integration** — VFX industry standard
11. **Plugin architecture** — ecosystem growth

---

## Where rasmcore Already Exceeds Competitors

1. **Portability** — No competitor runs as WASM in the browser with full GPU compute
2. **ACES pipeline completeness** — IDT/RRT/ODT + ACEScct grading, ahead of PS/LR/Affinity
3. **Analysis infrastructure** — Zero-readback GPU analysis chains, cross-node buffer sharing
4. **Retinex variants** — SSR/MSR/MSRCR not available in most competitors
5. **Morphology depth** — Full suite including skeletonize (Zhang-Suen GPU)
6. **f32 everywhere** — No 8-bit intermediate quantization unlike GIMP, IM
7. **Professional scopes** — Waveform, vectorscope, parade matching DaVinci Resolve
8. **CLF support** — Common LUT Format support ahead of most tools
9. **Content-addressed caching** — blake3 hash chains for intelligent cache reuse
10. **Frequency separation as a filter** — Directly available, not a manual multi-step process
