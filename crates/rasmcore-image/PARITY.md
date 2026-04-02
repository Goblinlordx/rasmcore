# Parity Scorecard — rasmcore vs OpenCV/ImageMagick Reference

Last updated: 2026-04-02

All results from automated parity tests on 7 canonical 128x128 grayscale images
(gradient, checker, noisy_flat, sharp_edges, photo, flat, highcontrast).

## Byte-Exact Filters (MAE=0.0000, max_err=0)

These filters produce identical output to OpenCV/ImageMagick on every pixel:

| Filter | Reference | Images | Status |
|--------|-----------|--------|--------|
| **Bilateral** | OpenCV | 7/7 exact | Byte-exact |
| **Bokeh disc r=3** | OpenCV filter2D | 7/7 exact | Byte-exact |
| **Bokeh disc r=7** | OpenCV filter2D | 7/7 exact | Byte-exact |
| **Bokeh hex r=3** | OpenCV filter2D | 7/7 exact | Byte-exact |
| **Bokeh hex r=7** | OpenCV filter2D | 7/7 exact | Byte-exact |
| **Displacement map (barrel)** | OpenCV remap | 7/7 exact | Byte-exact |
| **Displacement map (wave)** | OpenCV remap | 7/7 exact | Byte-exact |
| **Scharr** | OpenCV | 4/4 exact | Byte-exact |
| **Laplacian** | OpenCV | 4/4 exact | Byte-exact |
| **Morphology (all ops)** | OpenCV | 54/54 exact | Byte-exact |
| **Perspective warp** | OpenCV | 1/1 exact | Byte-exact |
| **pyrUp** | OpenCV | 7/7 exact | Byte-exact |
| **Gray-world WB** | Reference | 1/1 exact | Byte-exact |
| **Colorize** | W3C/Photoshop spec | 7/7 exact | W3C SetLum/ClipColor, BT.601 luma |
| **White balance temp** | ImageMagick -evaluate | 1/1 exact | Byte-exact channel multiply |
| **Photo filter** | ImageMagick -colorize | 1/1 exact | Byte-exact color blend |

## Near-Exact Filters (max_err <= 1)

| Filter | Reference | Worst MAE | Worst max_err | Notes |
|--------|-----------|-----------|---------------|-------|
| **CLAHE** | OpenCV | 0.094 | 1 | Rounding in histogram bin mapping |
| **Guided filter** | OpenCV | 0.007 | 1 | Floating-point accumulation order |

## Close Filters (MAE < 10)

| Filter | Reference | Worst MAE | Worst max_err | Notes |
|--------|-----------|-----------|---------------|-------|
| **Grayscale (BT.709)** | ImageMagick -fx | 0.50 | — | f32 rounding vs IM double precision |
| **Hue rotate** | ImageMagick -modulate | < 5.0 | — | HSL model differences at 270° |
| **Gradient map** | ImageMagick -clut | 0.003 | — | Near-exact against IM CLUT pipeline |
| **Canny** | OpenCV | 5.65 | 255 | f32 vs int16 tangent precision in NMS; 4/7 byte-exact |
| **Vignette** | ImageMagick | 1.26 | 5 | Different radial falloff formula |
| **Mertens fusion** | OpenCV | 5.76 (u8) | 29 | Checker image at exposure extremes; f32 MAE=0.024 |

## Codec Parity

| Codec | Direction | Reference | Status |
|-------|-----------|-----------|--------|
| **JPEG** | Decode | zune-jpeg/ImageMagick | Byte-exact (all modes including CMYK/YCCK) |
| **JPEG** | Encode | (our own round-trip) | Deterministic, validated |
| **PNG** | Encode | ImageMagick | Byte-exact |
| **BMP** | Both | image crate | Byte-exact (18 parity tests) |
| **QOI** | Both | image crate | Byte-exact (5 parity tests) |
| **TGA** | Both | image crate | Byte-exact (4 parity tests) |
| **PNM** | Both | image crate | Byte-exact (4 parity tests) |

## Summary

- **16 filter operations** are byte-exact against OpenCV/ImageMagick/W3C reference
- **2 filter operations** within max_err=1 (CLAHE, guided filter)
- **6 filter operations** have documented divergence with root causes
- **7 codec paths** validated with parity tests
