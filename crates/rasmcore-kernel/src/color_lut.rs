//! 3D color lookup table infrastructure — CLUT types, composition, and parsing.
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
//!
//! # External LUT Import
//!
//! Supports loading 3D LUTs from:
//! - **Adobe/Resolve .cube** files via [`parse_cube_lut`]

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

// ─── External LUT Import (.cube) ─────────────────────────────────────────

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
pub fn parse_cube_lut(text: &str) -> Result<ColorLut3D, String> {
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
        "missing LUT_3D_SIZE in .cube file".to_string()
    })?;

    let expected = n * n * n;
    if data.len() != expected {
        return Err(format!(
            ".cube LUT_3D_SIZE={n} expects {expected} entries, got {}",
            data.len()
        ));
    }

    Ok(ColorLut3D {
        grid_size: n,
        data,
    })
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

    // ── Identity LUT ────────────────────────────────────────────────────

    #[test]
    fn identity_lut_preserves_colors() {
        let lut = ColorLut3D::identity(33);
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
        let lut = ColorLut3D::from_fn(33, |r, g, b| (r * 0.8, g * 0.6, b * 0.4));

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
        assert!(
            max_err < 0.005,
            "tetrahedral max error {max_err} too high for linear transform"
        );
    }

    #[test]
    fn tetrahedral_nonlinear_within_tolerance() {
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
        assert!(
            max_err < 0.005,
            "tetrahedral max error {max_err} for gamma 2.2"
        );
    }

    // ── Composition ─────────────────────────────────────────────────────

    #[test]
    fn compose_cluts_matches_sequential() {
        let hue = ColorOp::HueRotate(90.0).to_clut(17);
        let sat = ColorOp::Saturate(0.5).to_clut(17);
        let fused = compose_cluts(&hue, &sat);

        let mut max_err: f32 = 0.0;
        for i in 0..500 {
            let r = (i * 37 % 256) as f32 / 255.0;
            let g = (i * 73 % 256) as f32 / 255.0;
            let b = (i * 131 % 256) as f32 / 255.0;

            let (r1, g1, b1) = hue.lookup(r, g, b);
            let (rs, gs, bs) = sat.lookup(r1, g1, b1);

            let (rf, gf, bf) = fused.lookup(r, g, b);

            max_err = max_err
                .max((rf - rs).abs())
                .max((gf - gs).abs())
                .max((bf - bs).abs());
        }
        assert!(
            max_err < 0.02,
            "compose max error {max_err} between fused and sequential"
        );
    }

    // ── .cube parsing ───────────────────────────────────────────────────

    #[test]
    fn parse_cube_identity_2x2x2() {
        let cube = "\
LUT_3D_SIZE 2
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
0.0 0.0 1.0
1.0 0.0 1.0
0.0 1.0 1.0
1.0 1.0 1.0
";
        let lut = parse_cube_lut(cube).unwrap();
        assert_eq!(lut.grid_size, 2);
        assert_eq!(lut.data.len(), 8);

        // Identity check at corners
        let (r, g, b) = lut.lookup(0.0, 0.0, 0.0);
        assert!((r).abs() < 1e-5);
        assert!((g).abs() < 1e-5);
        assert!((b).abs() < 1e-5);

        let (r, g, b) = lut.lookup(1.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 1e-5);
        assert!((g - 1.0).abs() < 1e-5);
        assert!((b - 1.0).abs() < 1e-5);
    }

    #[test]
    fn parse_cube_missing_size_errors() {
        let cube = "# just a comment\n0.0 0.0 0.0\n";
        assert!(parse_cube_lut(cube).is_err());
    }

    #[test]
    fn parse_cube_wrong_count_errors() {
        let cube = "LUT_3D_SIZE 2\n0.0 0.0 0.0\n";
        let err = parse_cube_lut(cube).unwrap_err();
        assert!(err.contains("expects 8 entries, got 1"));
    }
}
