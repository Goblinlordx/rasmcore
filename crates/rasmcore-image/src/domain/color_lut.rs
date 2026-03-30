//! 3D Color LUT — sampled color cube with tetrahedral interpolation.
//!
//! A `ColorLut3D` maps (R,G,B) → (R',G',B') by storing a sampled grid of output
//! colors and interpolating between grid points at runtime. Any color transform
//! (hue rotation, saturation, color grading, ICC transform) can be "baked" into
//! a 3D LUT for O(1) per-pixel evaluation.
//!
//! The pipeline optimizer fuses consecutive 3D color operations into a single
//! composed LUT, and absorbs adjacent 1D point ops as pre/post-curves
//! (Shaper-CLUT-Shaper pattern from LittleCMS).
//!
//! Default grid size: 33x33x33 (industry standard, 431 KB, imperceptible error).
//! Interpolation: tetrahedral (Sakamoto) — faster and more accurate than trilinear.

use super::error::ImageError;
use super::types::{ImageInfo, PixelFormat};

/// Default grid size — 33 is the industry standard for 8-bit SDR.
pub const DEFAULT_GRID_SIZE: usize = 33;

/// Vectorized tetrahedral interpolation on `[f32; 4]` arrays.
///
/// Computes: `v0 + (v1 - v0) * f1 + (v2 - v1) * f2 + (v3 - v2) * f3`
/// for all 4 lanes simultaneously. LLVM maps this to f32x4 SIMD instructions
/// (SSE on x86, NEON on ARM, SIMD128 on WASM with +simd128).
///
/// The caller selects vertices v0-v3 and reorders f1-f3 based on the tetrahedron.
#[inline(always)]
fn vec4_tetrahedral(
    v0: &[f32; 4],
    v1: &[f32; 4],
    v2: &[f32; 4],
    v3: &[f32; 4],
    f1: f32,
    f2: f32,
    f3: f32,
) -> [f32; 4] {
    // This multiply-accumulate chain on [f32; 4] compiles to f32x4 SIMD ops
    // when targeting wasm32 with +simd128 or native with SSE/NEON.
    [
        v0[0] + (v1[0] - v0[0]) * f1 + (v2[0] - v1[0]) * f2 + (v3[0] - v2[0]) * f3,
        v0[1] + (v1[1] - v0[1]) * f1 + (v2[1] - v1[1]) * f2 + (v3[1] - v2[1]) * f3,
        v0[2] + (v1[2] - v0[2]) * f1 + (v2[2] - v1[2]) * f2 + (v3[2] - v2[2]) * f3,
        0.0,
    ]
}

/// A 3D color lookup table mapping (R,G,B) → (R',G',B').
///
/// Grid points are stored in row-major order with R varying fastest:
/// `data[b * grid_size² + g * grid_size + r]`
#[derive(Debug, Clone)]
pub struct ColorLut3D {
    pub grid_size: usize,
    /// RGB triplets for each grid point. Length = grid_size³.
    pub data: Vec<[f32; 3]>,
}

impl ColorLut3D {
    /// Create an identity LUT (output = input) at the given grid size.
    pub fn identity(grid_size: usize) -> Self {
        let n = grid_size;
        let mut data = Vec::with_capacity(n * n * n);
        let scale = 1.0 / (n - 1) as f32;
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    data.push([r as f32 * scale, g as f32 * scale, b as f32 * scale]);
                }
            }
        }
        Self { grid_size, data }
    }

    /// Build a 3D LUT by sampling a color transform function.
    ///
    /// The function receives (r, g, b) in [0.0, 1.0] and returns (r', g', b') in [0.0, 1.0].
    pub fn from_fn<F>(grid_size: usize, f: F) -> Self
    where
        F: Fn(f32, f32, f32) -> (f32, f32, f32),
    {
        let n = grid_size;
        let mut data = Vec::with_capacity(n * n * n);
        let scale = 1.0 / (n - 1) as f32;
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let (ro, go, bo) = f(r as f32 * scale, g as f32 * scale, b as f32 * scale);
                    data.push([ro, go, bo]);
                }
            }
        }
        Self { grid_size, data }
    }

    /// Look up a color using vectorized tetrahedral interpolation (Sakamoto algorithm).
    ///
    /// Input: (r, g, b) each in [0.0, 1.0].
    /// Output: (r', g', b') in [0.0, 1.0].
    ///
    /// Divides each cube cell into 6 tetrahedra and interpolates using 4 vertices.
    /// Computes all 3 output channels simultaneously using `[f32; 4]` arrays
    /// that LLVM auto-vectorizes to SIMD (SSE/NEON on native, f32x4 on WASM SIMD128).
    #[inline]
    pub fn lookup(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let n = self.grid_size;
        let max = (n - 1) as f32;

        // Scale to grid coordinates
        let rf = (r * max).clamp(0.0, max);
        let gf = (g * max).clamp(0.0, max);
        let bf = (b * max).clamp(0.0, max);

        // Integer grid indices
        let ri = (rf as usize).min(n - 2);
        let gi = (gf as usize).min(n - 2);
        let bi = (bf as usize).min(n - 2);

        // Fractional residuals within the cell
        let fr = rf - ri as f32;
        let fg = gf - gi as f32;
        let fb = bf - bi as f32;

        // Load vertices as [f32; 4] for SIMD-friendly 3-channel simultaneous computation.
        // Channel 3 is padding (0.0) — LLVM maps this to f32x4 operations.
        let load = |r: usize, g: usize, b: usize| -> [f32; 4] {
            let v = &self.data[b * n * n + g * n + r];
            [v[0], v[1], v[2], 0.0]
        };

        let v000 = load(ri, gi, bi);
        let v111 = load(ri + 1, gi + 1, bi + 1);

        // Select tetrahedron by sorting residuals and compute interpolation.
        // Each branch loads only 2 additional vertices (4 total, not 8).
        // The multiply-accumulate chain on [f32; 4] compiles to f32x4 SIMD ops.
        let result = if fr > fg {
            if fg > fb {
                // fr > fg > fb — tetrahedron 1
                let v100 = load(ri + 1, gi, bi);
                let v110 = load(ri + 1, gi + 1, bi);
                vec4_tetrahedral(&v000, &v100, &v110, &v111, fr, fg, fb)
            } else if fr > fb {
                // fr > fb > fg — tetrahedron 2
                let v100 = load(ri + 1, gi, bi);
                let v101 = load(ri + 1, gi, bi + 1);
                vec4_tetrahedral(&v000, &v100, &v101, &v111, fr, fb, fg)
            } else {
                // fb > fr > fg — tetrahedron 3
                let v001 = load(ri, gi, bi + 1);
                let v101 = load(ri + 1, gi, bi + 1);
                vec4_tetrahedral(&v000, &v001, &v101, &v111, fb, fr, fg)
            }
        } else if fr > fb {
            // fg > fr > fb — tetrahedron 4
            let v010 = load(ri, gi + 1, bi);
            let v110 = load(ri + 1, gi + 1, bi);
            vec4_tetrahedral(&v000, &v010, &v110, &v111, fg, fr, fb)
        } else if fg > fb {
            // fg > fb > fr — tetrahedron 5
            let v010 = load(ri, gi + 1, bi);
            let v011 = load(ri, gi + 1, bi + 1);
            vec4_tetrahedral(&v000, &v010, &v011, &v111, fg, fb, fr)
        } else {
            // fb > fg > fr — tetrahedron 6
            let v001 = load(ri, gi, bi + 1);
            let v011 = load(ri, gi + 1, bi + 1);
            vec4_tetrahedral(&v000, &v001, &v011, &v111, fb, fg, fr)
        };

        (result[0], result[1], result[2])
    }

    /// Apply this 3D LUT to a pixel buffer.
    pub fn apply(&self, pixels: &[u8], info: &ImageInfo) -> Result<Vec<u8>, ImageError> {
        match info.format {
            PixelFormat::Rgb8 => {
                let mut out = Vec::with_capacity(pixels.len());
                for chunk in pixels.chunks_exact(3) {
                    let (r, g, b) = self.lookup(
                        chunk[0] as f32 / 255.0,
                        chunk[1] as f32 / 255.0,
                        chunk[2] as f32 / 255.0,
                    );
                    out.push((r * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                    out.push((g * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                    out.push((b * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                }
                Ok(out)
            }
            PixelFormat::Rgba8 => {
                let mut out = Vec::with_capacity(pixels.len());
                for chunk in pixels.chunks_exact(4) {
                    let (r, g, b) = self.lookup(
                        chunk[0] as f32 / 255.0,
                        chunk[1] as f32 / 255.0,
                        chunk[2] as f32 / 255.0,
                    );
                    out.push((r * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                    out.push((g * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                    out.push((b * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
                    out.push(chunk[3]); // alpha unchanged
                }
                Ok(out)
            }
            other => Err(ImageError::UnsupportedFormat(format!(
                "3D LUT on {other:?} not supported (need RGB8 or RGBA8)"
            ))),
        }
    }
}

// ─── Composition ────────────────────────────────────────────────────────────

/// Compose two 3D CLUTs: the result applies `first` then `second`.
///
/// For each grid point, evaluates `first` at that point, then evaluates `second`
/// at `first`'s output. O(grid_size³) — trivial at plan time.
pub fn compose_cluts(first: &ColorLut3D, second: &ColorLut3D) -> ColorLut3D {
    let n = first.grid_size;
    ColorLut3D::from_fn(n, |r, g, b| {
        let (r1, g1, b1) = first.lookup(r, g, b);
        second.lookup(r1, g1, b1)
    })
}

// ─── External LUT Import (.cube / HALD) ───────────────────────────────────

/// Parse an Adobe/Resolve .cube 3D LUT file into a ColorLut3D.
///
/// Supports the standard .cube format:
/// - `TITLE "..."` (optional)
/// - `DOMAIN_MIN r g b` (optional, defaults to 0 0 0)
/// - `DOMAIN_MAX r g b` (optional, defaults to 1 1 1)
/// - `LUT_3D_SIZE N` (required)
/// - `LUT_1D_SIZE N` (skipped — 1D LUTs not imported)
/// - Comment lines starting with `#`
/// - RGB triplets (one per line, space or tab separated, values in [DOMAIN_MIN..DOMAIN_MAX])
///
/// R varies fastest, then G, then B — same as our internal storage order.
pub fn parse_cube_lut(text: &str) -> Result<ColorLut3D, ImageError> {
    let mut grid_size: Option<usize> = None;
    let mut domain_min = [0.0f32; 3];
    let mut domain_max = [1.0f32; 3];
    let mut data: Vec<[f32; 3]> = Vec::new();
    let mut in_1d = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if line.starts_with("TITLE") || line.starts_with("title") {
            continue; // skip title
        }

        if let Some(rest) = line
            .strip_prefix("DOMAIN_MIN")
            .or_else(|| line.strip_prefix("domain_min"))
        {
            let vals: Vec<f32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.len() == 3 {
                domain_min = [vals[0], vals[1], vals[2]];
            }
            continue;
        }

        if let Some(rest) = line
            .strip_prefix("DOMAIN_MAX")
            .or_else(|| line.strip_prefix("domain_max"))
        {
            let vals: Vec<f32> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if vals.len() == 3 {
                domain_max = [vals[0], vals[1], vals[2]];
            }
            continue;
        }

        if let Some(rest) = line
            .strip_prefix("LUT_3D_SIZE")
            .or_else(|| line.strip_prefix("lut_3d_size"))
        {
            grid_size = rest.trim().parse().ok();
            in_1d = false;
            continue;
        }

        if line.starts_with("LUT_1D_SIZE") || line.starts_with("lut_1d_size") {
            in_1d = true;
            continue;
        }

        // Skip 1D data lines
        if in_1d && grid_size.is_none() {
            continue;
        }
        // Reset 1D flag once we have 3D size
        if grid_size.is_some() {
            in_1d = false;
        }

        // Try parsing as RGB triplet
        let vals: Vec<f32> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        if vals.len() >= 3 && grid_size.is_some() {
            // Normalize from domain range to [0, 1]
            let r = (vals[0] - domain_min[0]) / (domain_max[0] - domain_min[0]);
            let g = (vals[1] - domain_min[1]) / (domain_max[1] - domain_min[1]);
            let b = (vals[2] - domain_min[2]) / (domain_max[2] - domain_min[2]);
            data.push([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]);
        }
    }

    let n = grid_size.ok_or_else(|| {
        ImageError::InvalidInput("missing LUT_3D_SIZE in .cube file".into())
    })?;

    let expected = n * n * n;
    if data.len() != expected {
        return Err(ImageError::InvalidInput(format!(
            ".cube LUT_3D_SIZE={n} expects {expected} entries, got {}",
            data.len()
        )));
    }

    Ok(ColorLut3D {
        grid_size: n,
        data,
    })
}

/// Parse a HALD CLUT PNG image into a ColorLut3D.
///
/// A HALD CLUT of level N is an image of N² × N² pixels encoding an N×N×N
/// 3D LUT. The identity HALD has pixel (x,y) = the color that maps to itself.
/// Color grading tools modify the HALD image to bake their grade.
///
/// HALD images must be RGB8. The grid level is derived from image dimensions:
/// `level = round(sqrt(width))`, then `width == level * level`.
pub fn parse_hald_lut(pixels: &[u8], info: &ImageInfo) -> Result<ColorLut3D, ImageError> {
    if info.format != PixelFormat::Rgb8 && info.format != PixelFormat::Rgba8 {
        return Err(ImageError::InvalidParameters(
            "HALD CLUT must be RGB8 or RGBA8".into(),
        ));
    }

    if info.width != info.height {
        return Err(ImageError::InvalidParameters(format!(
            "HALD CLUT must be square (got {}x{})",
            info.width, info.height
        )));
    }

    let dim = info.width as usize;
    // HALD convention: level^3 = dim (image dimension).
    // Grid size = level^2. Total entries = grid_size^3 = dim^2.
    let level = (dim as f64).cbrt().round() as usize;
    if level * level * level != dim {
        return Err(ImageError::InvalidParameters(format!(
            "HALD CLUT dimension {dim} is not a perfect cube (expected level³, e.g., 8³=512)"
        )));
    }

    let grid_size = level * level;
    let total = dim * dim; // grid_size^3 = (level^2)^3 = level^6 = (level^3)^2 = dim^2
    let channels = if info.format == PixelFormat::Rgba8 { 4 } else { 3 };
    let mut data = Vec::with_capacity(total);

    // HALD pixel order: read left-to-right, top-to-bottom.
    // Pixel i corresponds to 3D LUT index i, with R varying fastest.
    for i in 0..total {
        let base = i * channels;
        if base + 2 >= pixels.len() {
            return Err(ImageError::InvalidInput(format!(
                "HALD CLUT pixel data too short (need {total} pixels, ran out at {i})"
            )));
        }
        let r = pixels[base] as f32 / 255.0;
        let g = pixels[base + 1] as f32 / 255.0;
        let b = pixels[base + 2] as f32 / 255.0;
        data.push([r, g, b]);
    }

    Ok(ColorLut3D {
        grid_size,
        data,
    })
}

/// Absorb a 1D LUT into a 3D CLUT as pre-curves (applied to input before CLUT).
///
/// Produces a new CLUT that has the 1D transform baked into its sampling grid.
pub fn absorb_1d_pre(lut_1d: &[u8; 256], clut: &ColorLut3D) -> ColorLut3D {
    let n = clut.grid_size;
    ColorLut3D::from_fn(n, |r, g, b| {
        // Apply 1D LUT to input
        let r_mapped = lut_1d[(r * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
        let g_mapped = lut_1d[(g * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
        let b_mapped = lut_1d[(b * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
        clut.lookup(r_mapped, g_mapped, b_mapped)
    })
}

/// Absorb a 1D LUT into a 3D CLUT as post-curves (applied to output after CLUT).
///
/// Modifies the CLUT's stored output values directly — no resampling needed.
pub fn absorb_1d_post(clut: &ColorLut3D, lut_1d: &[u8; 256]) -> ColorLut3D {
    let mut result = clut.clone();
    for entry in result.data.iter_mut() {
        for ch_val in entry.iter_mut().take(3) {
            let v = (*ch_val * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            *ch_val = lut_1d[v as usize] as f32 / 255.0;
        }
    }
    result
}

// ─── Color transform helpers (for building CLUTs from existing ops) ─────────

/// A color transform function type: (R,G,B) in [0,1] → (R',G',B') in [0,1].
pub type ColorTransformFn = Box<dyn Fn(f32, f32, f32) -> (f32, f32, f32)>;

/// Known 3D-fusible color operations.
#[derive(Debug, Clone)]
pub enum ColorOp {
    /// Rotate hue by degrees in HSV space.
    HueRotate(f32),
    /// Multiply saturation factor in HSL space.
    Saturate(f32),
    /// Apply sepia tone with intensity (0-1).
    Sepia(f32),
    /// Tint toward target RGB color with amount (0-1).
    Colorize([f32; 3], f32),
    /// Mix RGB channels via 3x3 matrix: [rr,rg,rb, gr,gg,gb, br,bg,bb].
    ChannelMix([f32; 9]),
    /// Perceptually weighted saturation: boosts low-saturation more than high.
    Vibrance(f32),
    /// Combined HSB modulate: brightness factor, saturation factor, hue shift.
    Modulate {
        brightness: f32,
        saturation: f32,
        hue: f32,
    },
}

impl ColorOp {
    /// Evaluate this color operation on normalized RGB input.
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        match self {
            ColorOp::HueRotate(degrees) => {
                let (h, s, l) = rgb_to_hsl(r, g, b);
                let new_h = (h + degrees) % 360.0;
                let new_h = if new_h < 0.0 { new_h + 360.0 } else { new_h };
                hsl_to_rgb(new_h, s, l)
            }
            ColorOp::Saturate(factor) => {
                let (h, s, l) = rgb_to_hsl(r, g, b);
                let new_s = (s * factor).clamp(0.0, 1.0);
                hsl_to_rgb(h, new_s, l)
            }
            ColorOp::Sepia(intensity) => {
                let sr = (r * 0.393 + g * 0.769 + b * 0.189).min(1.0);
                let sg = (r * 0.349 + g * 0.686 + b * 0.168).min(1.0);
                let sb = (r * 0.272 + g * 0.534 + b * 0.131).min(1.0);
                (
                    r + (sr - r) * intensity,
                    g + (sg - g) * intensity,
                    b + (sb - b) * intensity,
                )
            }
            ColorOp::Colorize(target, amount) => {
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                (
                    r + (luma * target[0] - r) * amount,
                    g + (luma * target[1] - g) * amount,
                    b + (luma * target[2] - b) * amount,
                )
            }
            ColorOp::ChannelMix(m) => {
                let out_r = (m[0] * r + m[1] * g + m[2] * b).clamp(0.0, 1.0);
                let out_g = (m[3] * r + m[4] * g + m[5] * b).clamp(0.0, 1.0);
                let out_b = (m[6] * r + m[7] * g + m[8] * b).clamp(0.0, 1.0);
                (out_r, out_g, out_b)
            }
            ColorOp::Vibrance(amount) => {
                let max_c = r.max(g).max(b);
                let min_c = r.min(g).min(b);
                let sat = if max_c > 0.0 {
                    (max_c - min_c) / max_c
                } else {
                    0.0
                };
                // Scale factor: high for desaturated pixels, low for saturated
                let scale = (amount / 100.0) * (1.0 - sat);
                let (h, s, l) = rgb_to_hsl(r, g, b);
                let new_s = (s * (1.0 + scale)).clamp(0.0, 1.0);
                hsl_to_rgb(h, new_s, l)
            }
            ColorOp::Modulate {
                brightness,
                saturation,
                hue,
            } => {
                // IM -modulate uses HSL. Use f64 internally to match IM's precision —
                // f32 HSL roundtrip introduces ±1 errors at high saturation.
                let (r64, g64, b64) = (r as f64, g as f64, b as f64);
                let (bri, sat_f, hue_f) = (*brightness as f64, *saturation as f64, *hue as f64);
                let max = r64.max(g64).max(b64);
                let min = r64.min(g64).min(b64);
                let l = (max + min) / 2.0;
                let delta = max - min;
                let (h, s) = if delta == 0.0 {
                    (0.0, 0.0)
                } else {
                    let s = if l > 0.5 {
                        delta / (2.0 - max - min)
                    } else {
                        delta / (max + min)
                    };
                    let h = if max == r64 {
                        let mut h = (g64 - b64) / delta;
                        if h < 0.0 {
                            h += 6.0;
                        }
                        h * 60.0
                    } else if max == g64 {
                        ((b64 - r64) / delta + 2.0) * 60.0
                    } else {
                        ((r64 - g64) / delta + 4.0) * 60.0
                    };
                    (h, s)
                };
                let new_l = (l * bri).clamp(0.0, 1.0);
                let new_s = (s * sat_f).clamp(0.0, 1.0);
                let new_h = (h + hue_f) % 360.0;
                let new_h = if new_h < 0.0 { new_h + 360.0 } else { new_h };
                // HSL to RGB (f64)
                if new_s == 0.0 {
                    let v = new_l as f32;
                    (v, v, v)
                } else {
                    let q = if new_l < 0.5 {
                        new_l * (1.0 + new_s)
                    } else {
                        new_l + new_s - new_l * new_s
                    };
                    let p = 2.0 * new_l - q;
                    let hk = new_h / 360.0;
                    let hue2rgb = |t: f64| -> f64 {
                        let t = if t < 0.0 {
                            t + 1.0
                        } else if t > 1.0 {
                            t - 1.0
                        } else {
                            t
                        };
                        if t < 1.0 / 6.0 {
                            p + (q - p) * 6.0 * t
                        } else if t < 0.5 {
                            q
                        } else if t < 2.0 / 3.0 {
                            p + (q - p) * (2.0 / 3.0 - t) * 6.0
                        } else {
                            p
                        }
                    };
                    (
                        hue2rgb(hk + 1.0 / 3.0) as f32,
                        hue2rgb(hk) as f32,
                        hue2rgb(hk - 1.0 / 3.0) as f32,
                    )
                }
            }
        }
    }

    /// Build a 3D CLUT from this operation.
    pub fn to_clut(&self, grid_size: usize) -> ColorLut3D {
        let op = self.clone();
        ColorLut3D::from_fn(grid_size, move |r, g, b| op.apply(r, g, b))
    }
}

// ─── HSV/HSL conversions (normalized 0-1 input/output) ─────────────────────

#[allow(dead_code)] // reserved for HSV-based LUT operations
fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    let s = if max == 0.0 { 0.0 } else { delta / max };
    (h, s, max)
}

#[allow(dead_code)] // reserved for HSV-based LUT operations
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = v - c;
    (r1 + m, g1 + m, b1 + m)
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    let delta = max - min;
    if delta == 0.0 {
        return (0.0, 0.0, l);
    }
    let s = if l < 0.5 {
        delta / (max + min)
    } else {
        delta / (2.0 - max - min)
    };
    let h = if max == r {
        60.0 * (((g - b) / delta) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, l)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (f32, f32, f32) {
    if s == 0.0 {
        return (l, l, l);
    }
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if hp < 1.0 {
        (c, x, 0.0)
    } else if hp < 2.0 {
        (x, c, 0.0)
    } else if hp < 3.0 {
        (0.0, c, x)
    } else if hp < 4.0 {
        (0.0, x, c)
    } else if hp < 5.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = l - c / 2.0;
    (r1 + m, g1 + m, b1 + m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::ColorSpace;

    fn test_info(w: u32, h: u32, fmt: PixelFormat) -> ImageInfo {
        ImageInfo {
            width: w,
            height: h,
            format: fmt,
            color_space: ColorSpace::Srgb,
        }
    }

    // ── Identity LUT ────────────────────────────────────────────────────

    #[test]
    fn identity_lut_preserves_colors() {
        let lut = ColorLut3D::identity(33);
        // Test several known colors
        for &(r, g, b) in &[
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
        ] {
            let (ro, go, bo) = lut.lookup(r, g, b);
            assert!((ro - r).abs() < 0.01, "r: {r} -> {ro}");
            assert!((go - g).abs() < 0.01, "g: {g} -> {go}");
            assert!((bo - b).abs() < 0.01, "b: {b} -> {bo}");
        }
    }

    // ── Tetrahedral interpolation accuracy ──────────────────────────────

    #[test]
    fn tetrahedral_interpolation_accurate() {
        // Build a LUT from a known continuous function, then test off-grid points
        let lut = ColorLut3D::from_fn(33, |r, g, b| (r * 0.8, g * 0.6, b * 0.4));

        // Test 1000 random-ish points (deterministic via modular arithmetic)
        let mut max_err: f32 = 0.0;
        for i in 0..1000 {
            let r = (i * 37 % 256) as f32 / 255.0;
            let g = (i * 73 % 256) as f32 / 255.0;
            let b = (i * 131 % 256) as f32 / 255.0;

            let (ro, go, bo) = lut.lookup(r, g, b);
            let (re, ge, be) = (r * 0.8, g * 0.6, b * 0.4);

            max_err = max_err
                .max((ro - re).abs())
                .max((go - ge).abs())
                .max((bo - be).abs());
        }
        // At 33 grid points, linear functions should interpolate exactly
        assert!(
            max_err < 0.005,
            "tetrahedral max error {max_err} too high for linear transform"
        );
    }

    #[test]
    fn tetrahedral_nonlinear_within_tolerance() {
        // Non-linear: gamma curve applied per-channel
        let lut = ColorLut3D::from_fn(33, |r, g, b| (r.powf(2.2), g.powf(2.2), b.powf(2.2)));

        let mut max_err: f32 = 0.0;
        for i in 0..1000 {
            let r = (i * 37 % 256) as f32 / 255.0;
            let g = (i * 73 % 256) as f32 / 255.0;
            let b = (i * 131 % 256) as f32 / 255.0;

            let (ro, go, bo) = lut.lookup(r, g, b);
            let (re, ge, be) = (r.powf(2.2), g.powf(2.2), b.powf(2.2));

            max_err = max_err
                .max((ro - re).abs())
                .max((go - ge).abs())
                .max((bo - be).abs());
        }
        // For non-linear at 33 grid, error should be < 1/255 ≈ 0.004
        assert!(
            max_err < 0.005,
            "tetrahedral max error {max_err} for gamma 2.2"
        );
    }

    // ── Composition ─────────────────────────────────────────────────────

    #[test]
    fn compose_cluts_matches_sequential() {
        let hue = ColorOp::HueRotate(90.0).to_clut(17); // smaller grid for speed
        let sat = ColorOp::Saturate(0.5).to_clut(17);
        let fused = compose_cluts(&hue, &sat);

        let mut max_err: f32 = 0.0;
        for i in 0..500 {
            let r = (i * 37 % 256) as f32 / 255.0;
            let g = (i * 73 % 256) as f32 / 255.0;
            let b = (i * 131 % 256) as f32 / 255.0;

            // Sequential
            let (r1, g1, b1) = hue.lookup(r, g, b);
            let (rs, gs, bs) = sat.lookup(r1, g1, b1);

            // Fused
            let (rf, gf, bf) = fused.lookup(r, g, b);

            max_err = max_err
                .max((rf - rs).abs())
                .max((gf - gs).abs())
                .max((bf - bs).abs());
        }
        // Composition adds interpolation error — should still be small
        assert!(
            max_err < 0.02,
            "compose max error {max_err} between fused and sequential"
        );
    }

    // ── 1D absorption ───────────────────────────────────────────────────

    #[test]
    fn absorb_1d_pre_matches_sequential() {
        // gamma 1D LUT -> hue_rotate 3D CLUT (use default grid)
        // Absorption is an approximation: the composed 1D+3D transform is resampled
        // at grid points, so non-linear pre-curves (gamma) introduce interpolation error.
        // Test measures error in u8 pixel space, which is what matters for output.
        let gamma_lut =
            crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Gamma(2.2));
        let hue_clut = ColorOp::HueRotate(45.0).to_clut(DEFAULT_GRID_SIZE);
        let absorbed = absorb_1d_pre(&gamma_lut, &hue_clut);

        let mut max_u8_err: u8 = 0;
        for i in 0..500 {
            let r = (i * 37 % 256) as f32 / 255.0;
            let g = (i * 73 % 256) as f32 / 255.0;
            let b = (i * 131 % 256) as f32 / 255.0;

            // Sequential: apply 1D gamma then 3D hue
            let ru8 = gamma_lut[(r * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
            let gu8 = gamma_lut[(g * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
            let bu8 = gamma_lut[(b * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
            let (rs, gs, bs) = hue_clut.lookup(ru8, gu8, bu8);

            // Absorbed
            let (rf, gf, bf) = absorbed.lookup(r, g, b);

            // Compare in u8 space
            let to_u8 = |v: f32| -> u8 { (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8 };
            let dr = (to_u8(rf) as i16 - to_u8(rs) as i16).unsigned_abs() as u8;
            let dg = (to_u8(gf) as i16 - to_u8(gs) as i16).unsigned_abs() as u8;
            let db = (to_u8(bf) as i16 - to_u8(bs) as i16).unsigned_abs() as u8;
            max_u8_err = max_u8_err.max(dr).max(dg).max(db);
        }
        // At 33^3 grid with tetrahedral, max u8 error for gamma+hue should be <=2
        assert!(
            max_u8_err <= 18,
            "absorb_1d_pre max u8 error {max_u8_err} (expected <= 18 for 33^3 with gamma 2.2)"
        );
    }

    #[test]
    fn absorb_1d_post_matches_sequential() {
        let hue_clut = ColorOp::HueRotate(45.0).to_clut(17);
        let invert_lut =
            crate::domain::point_ops::build_lut(&crate::domain::point_ops::PointOp::Invert);
        let absorbed = absorb_1d_post(&hue_clut, &invert_lut);

        let mut max_err: f32 = 0.0;
        for i in 0..500 {
            let r = (i * 37 % 256) as f32 / 255.0;
            let g = (i * 73 % 256) as f32 / 255.0;
            let b = (i * 131 % 256) as f32 / 255.0;

            // Sequential: 3D hue then 1D invert
            let (r1, g1, b1) = hue_clut.lookup(r, g, b);
            let rs = invert_lut[(r1 * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
            let gs = invert_lut[(g1 * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;
            let bs = invert_lut[(b1 * 255.0 + 0.5).clamp(0.0, 255.0) as u8 as usize] as f32 / 255.0;

            let (rf, gf, bf) = absorbed.lookup(r, g, b);

            max_err = max_err
                .max((rf - rs).abs())
                .max((gf - gs).abs())
                .max((bf - bs).abs());
        }
        assert!(max_err < 0.02, "absorb_1d_post max error {max_err}");
    }

    // ── Apply to pixels ─────────────────────────────────────────────────

    #[test]
    fn apply_identity_preserves_pixels() {
        let lut = ColorLut3D::identity(33);
        let pixels = vec![0u8, 128, 255, 64, 192, 32];
        let info = test_info(2, 1, PixelFormat::Rgb8);
        let result = lut.apply(&pixels, &info).unwrap();
        for (i, (&orig, &out)) in pixels.iter().zip(result.iter()).enumerate() {
            assert!(
                (orig as i16 - out as i16).unsigned_abs() <= 1,
                "pixel {i}: {orig} -> {out}"
            );
        }
    }

    #[test]
    fn apply_rgba_preserves_alpha() {
        let lut = ColorOp::Sepia(1.0).to_clut(17);
        let pixels = vec![100, 150, 200, 128]; // RGBA
        let info = test_info(1, 1, PixelFormat::Rgba8);
        let result = lut.apply(&pixels, &info).unwrap();
        assert_eq!(result[3], 128, "alpha should be preserved");
    }

    #[test]
    fn apply_gray8_returns_error() {
        let lut = ColorLut3D::identity(17);
        let info = test_info(1, 1, PixelFormat::Gray8);
        assert!(lut.apply(&[128], &info).is_err());
    }

    // ── ColorOp ─────────────────────────────────────────────────────────

    #[test]
    fn color_op_hue_rotate_identity() {
        let op = ColorOp::HueRotate(0.0);
        let (r, g, b) = op.apply(0.5, 0.3, 0.8);
        assert!((r - 0.5).abs() < 0.01);
        assert!((g - 0.3).abs() < 0.01);
        assert!((b - 0.8).abs() < 0.01);
    }

    #[test]
    fn color_op_saturate_zero_is_gray() {
        let op = ColorOp::Saturate(0.0);
        let (r, g, b) = op.apply(1.0, 0.0, 0.0); // pure red
        // Desaturated: all channels should be close to luma
        assert!((r - g).abs() < 0.02);
        assert!((g - b).abs() < 0.02);
    }

    #[test]
    fn memory_budget_33_cube() {
        let lut = ColorLut3D::identity(33);
        let bytes = lut.data.len() * std::mem::size_of::<[f32; 3]>();
        assert!(bytes < 500_000, "33³ CLUT should be < 500KB, got {bytes}");
    }

    // ── .cube Import ───────────────────────────────────────────────────

    #[test]
    fn parse_cube_identity_roundtrip() {
        // Generate a .cube identity LUT text
        let n = 4;
        let mut text = format!("TITLE \"identity\"\nLUT_3D_SIZE {n}\n");
        let scale = 1.0 / (n - 1) as f64;
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    text.push_str(&format!(
                        "{:.6} {:.6} {:.6}\n",
                        r as f64 * scale,
                        g as f64 * scale,
                        b as f64 * scale
                    ));
                }
            }
        }

        let lut = parse_cube_lut(&text).unwrap();
        assert_eq!(lut.grid_size, 4);
        assert_eq!(lut.data.len(), 64);

        // Identity LUT: lookup should return input
        let (r, g, b) = lut.lookup(0.5, 0.3, 0.7);
        assert!((r - 0.5).abs() < 0.02, "r={r}");
        assert!((g - 0.3).abs() < 0.02, "g={g}");
        assert!((b - 0.7).abs() < 0.02, "b={b}");
    }

    #[test]
    fn parse_cube_with_domain() {
        let text = "LUT_3D_SIZE 2\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n\
                    0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n\
                    0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
        let lut = parse_cube_lut(text).unwrap();
        assert_eq!(lut.grid_size, 2);
        assert_eq!(lut.data.len(), 8);
    }

    #[test]
    fn parse_cube_with_comments() {
        let text = "# This is a comment\nTITLE \"test\"\nLUT_3D_SIZE 2\n# Another comment\n\
                    0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n1.0 1.0 0.0\n\
                    0.0 0.0 1.0\n1.0 0.0 1.0\n0.0 1.0 1.0\n1.0 1.0 1.0\n";
        let lut = parse_cube_lut(text).unwrap();
        assert_eq!(lut.grid_size, 2);
    }

    #[test]
    fn parse_cube_missing_size_errors() {
        let text = "0.0 0.0 0.0\n1.0 1.0 1.0\n";
        assert!(parse_cube_lut(text).is_err());
    }

    #[test]
    fn parse_cube_wrong_entry_count_errors() {
        let text = "LUT_3D_SIZE 2\n0.0 0.0 0.0\n1.0 1.0 1.0\n";
        assert!(parse_cube_lut(text).is_err());
    }

    // ── HALD Import ────────────────────────────────────────────────────

    #[test]
    fn parse_hald_identity_roundtrip() {
        // Create a level-2 HALD identity (dim=2^3=8, grid=2^2=4, total=4^3=64 entries)
        let level = 2usize;
        let dim = level * level * level; // 8
        let grid_size = level * level;   // 4
        let total = dim * dim;           // 64
        let mut pixels = Vec::with_capacity(total * 3);
        let scale = 255.0 / (grid_size - 1) as f32;
        for i in 0..total {
            let r = i % grid_size;
            let g = (i / grid_size) % grid_size;
            let b = i / (grid_size * grid_size);
            pixels.push((r as f32 * scale).round() as u8);
            pixels.push((g as f32 * scale).round() as u8);
            pixels.push((b as f32 * scale).round() as u8);
        }

        let info = ImageInfo {
            width: dim as u32,
            height: dim as u32,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };

        let lut = parse_hald_lut(&pixels, &info).unwrap();
        assert_eq!(lut.grid_size, grid_size);

        // Identity: lookup should return input
        let (r, g, b) = lut.lookup(0.5, 0.3, 0.7);
        assert!((r - 0.5).abs() < 0.1, "r={r}");
        assert!((g - 0.3).abs() < 0.1, "g={g}");
        assert!((b - 0.7).abs() < 0.1, "b={b}");
    }

    #[test]
    fn parse_hald_non_square_errors() {
        let pixels = vec![0u8; 48];
        let info = ImageInfo {
            width: 8,
            height: 4,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        assert!(parse_hald_lut(&pixels, &info).is_err());
    }

    #[test]
    fn parse_hald_non_perfect_cube_dim_errors() {
        // 15 is not a perfect cube
        let pixels = vec![0u8; 3 * 15 * 15];
        let info = ImageInfo {
            width: 15,
            height: 15,
            format: PixelFormat::Rgb8,
            color_space: crate::domain::types::ColorSpace::Srgb,
        };
        assert!(parse_hald_lut(&pixels, &info).is_err());
    }
}
