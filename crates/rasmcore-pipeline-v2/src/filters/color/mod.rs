//! Color filters — f32-only color operations.
//!
//! Three categories:
//! 1. **CLUT-compatible**: Per-pixel RGB→RGB transforms that can be baked into
//!    a Clut3D for fusion (hue_rotate, saturate, channel_mixer, etc.)
//! 2. **Image-dependent**: Need full-image statistics (gray world WB, gradient map,
//!    quantize, dither) — not fusable via CLUT.
//! 3. **Spatial**: Need pixel neighborhoods (lab_sharpen, sparse_color).
//!
//! CLUT-compatible filters implement `ClutOp` to expose their Clut3D for the
//! fusion optimizer. The optimizer composes consecutive ClutOps into a single
//! fused CLUT pass.

use crate::fusion::Clut3D;

pub mod aces;
pub mod channel_mixer;
pub mod colorize;
pub mod dither_floyd_steinberg;
pub mod dither_ordered;
pub mod gradient_map;
pub mod hue_rotate;
pub mod kmeans_quantize;
pub mod lab_adjust;
pub mod lab_sharpen;
pub mod modulate;
pub mod photo_filter;
pub mod quantize;
pub mod replace_color;
pub mod saturate;
pub mod saturate_hsl;
pub mod selective_color;
pub mod sepia;
pub mod sparse_color;
pub mod vibrance;
pub mod white_balance_gray_world;
pub mod white_balance_temperature;

pub use aces::{AcesCctToCg, AcesCgToCct, AcesIdt, AcesOdt};
pub use channel_mixer::ChannelMixer;
pub use colorize::Colorize;
pub use dither_floyd_steinberg::DitherFloydSteinberg;
pub use dither_ordered::DitherOrdered;
pub use gradient_map::GradientMap;
pub use hue_rotate::HueRotate;
pub use kmeans_quantize::KmeansQuantize;
pub use lab_adjust::LabAdjust;
pub use lab_sharpen::LabSharpen;
pub use modulate::Modulate;
pub use photo_filter::PhotoFilter;
pub use quantize::Quantize;
pub use replace_color::ReplaceColor;
pub use saturate::Saturate;
pub use saturate_hsl::SaturateHsl;
pub use selective_color::SelectiveColor;
pub use sepia::Sepia;
pub use sparse_color::SparseColor;
pub use vibrance::Vibrance;
pub use white_balance_gray_world::WhiteBalanceGrayWorld;
pub use white_balance_temperature::WhiteBalanceTemperature;

// ─── ClutOp Trait ──────────────────────────────────────────────────────────

/// Color operation that can be represented as a 3D CLUT for fusion.
///
/// Filters implementing this trait can be composed by the fusion optimizer
/// into a single Clut3D pass via `compose_cluts()`.
pub trait ClutOp {
    /// Build a Clut3D representing this color operation.
    /// Grid size 33 is standard (33^3 ≈ 36K entries, good quality/size tradeoff).
    fn build_clut(&self) -> Clut3D;
}

// ─── Shared Color Space Helpers ────────────────────────────────────────────

/// Linear RGB → CIE Lab (L in [0,100], a,b in ~[-128,127]).
/// Input must be linear f32 (pipeline working space). D65 whitepoint.
pub(crate) fn rgb_to_lab(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    // Linear RGB → XYZ (D65, IEC 61966-2-1)
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.119192 * g + 0.9503041 * b;
    // XYZ → Lab (D65 white point)
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;
    let fx = lab_f(x / xn);
    let fy = lab_f(y / yn);
    let fz = lab_f(z / zn);
    let l_star = 116.0 * fy - 16.0;
    let a_star = 500.0 * (fx - fy);
    let b_star = 200.0 * (fy - fz);
    (l_star, a_star, b_star)
}

/// CIE Lab → linear RGB. Output is linear f32 (pipeline working space).
pub(crate) fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;
    let xn = 0.95047_f32;
    let yn = 1.0_f32;
    let zn = 1.08883_f32;
    let x = xn * lab_f_inv(fx);
    let y = yn * lab_f_inv(fy);
    let z = zn * lab_f_inv(fz);
    // XYZ → linear RGB (IEC 61966-2-1 inverse)
    let rl = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gl = -0.969266 * x + 1.8760108 * y + 0.041556 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (rl, gl, bl)
}

fn lab_f(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta * delta * delta {
        t.cbrt()
    } else {
        t / (3.0 * delta * delta) + 4.0 / 29.0
    }
}

fn lab_f_inv(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta {
        t * t * t
    } else {
        3.0 * delta * delta * (t - 4.0 / 29.0)
    }
}


// ─── Palette Helpers ──────────────────────────────────────────────────────

pub(crate) fn median_cut_palette(pixels: &[f32], max_colors: usize) -> Vec<(f32, f32, f32)> {
    // Collect unique-ish colors (subsample for speed on large images)
    let pixel_count = pixels.len() / 4;
    let step = (pixel_count / 4096).max(1);
    let mut colors: Vec<[f32; 3]> = Vec::with_capacity(pixel_count / step + 1);
    for i in (0..pixel_count).step_by(step) {
        let idx = i * 4;
        colors.push([pixels[idx], pixels[idx + 1], pixels[idx + 2]]);
    }
    if colors.is_empty() {
        return vec![(0.0, 0.0, 0.0)];
    }
    // Recursive median cut
    let mut buckets = vec![colors];
    while buckets.len() < max_colors {
        // Find bucket with largest range
        let (split_idx, _) = buckets
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let ra = color_range(a);
                let rb = color_range(b);
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let bucket = buckets.swap_remove(split_idx);
        if bucket.len() < 2 {
            buckets.push(bucket);
            break;
        }
        let channel = widest_channel(&bucket);
        let mut sorted = bucket;
        sorted.sort_by(|a, b| {
            a[channel]
                .partial_cmp(&b[channel])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mid = sorted.len() / 2;
        let (left, right) = sorted.split_at(mid);
        buckets.push(left.to_vec());
        buckets.push(right.to_vec());
    }
    buckets
        .iter()
        .map(|b| {
            let n = b.len() as f32;
            let (sr, sg, sb) = b.iter().fold((0.0, 0.0, 0.0), |(sr, sg, sb), c| {
                (sr + c[0], sg + c[1], sb + c[2])
            });
            (sr / n, sg / n, sb / n)
        })
        .collect()
}

fn color_range(colors: &[[f32; 3]]) -> f32 {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for c in colors {
        for i in 0..3 {
            min[i] = min[i].min(c[i]);
            max[i] = max[i].max(c[i]);
        }
    }
    (max[0] - min[0]).max(max[1] - min[1]).max(max[2] - min[2])
}

fn widest_channel(colors: &[[f32; 3]]) -> usize {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for c in colors {
        for i in 0..3 {
            min[i] = min[i].min(c[i]);
            max[i] = max[i].max(c[i]);
        }
    }
    let ranges = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    if ranges[0] >= ranges[1] && ranges[0] >= ranges[2] {
        0
    } else if ranges[1] >= ranges[2] {
        1
    } else {
        2
    }
}

pub(crate) fn nearest_color(
    palette: &[(f32, f32, f32)],
    r: f32,
    g: f32,
    b: f32,
) -> (f32, f32, f32) {
    let mut best = palette[0];
    let mut best_dist = f32::MAX;
    for &(pr, pg, pb) in palette {
        let dr = r - pr;
        let dg = g - pg;
        let db = b - pb;
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best = (pr, pg, pb);
        }
    }
    best
}

pub(crate) fn kmeans_palette(
    pixels: &[f32],
    k: usize,
    max_iterations: u32,
    seed: u32,
) -> Vec<(f32, f32, f32)> {
    use crate::noise;
    let pixel_count = pixels.len() / 4;
    if pixel_count == 0 || k == 0 {
        return vec![(0.0, 0.0, 0.0)];
    }
    // Initialize centroids via seeded selection (SplitMix64)
    let mut centroids: Vec<[f32; 3]> = Vec::with_capacity(k);
    let km_seed = seed as u64 ^ noise::SEED_KMEANS;
    for i in 0..k {
        let pi = noise::seeded_index(i as u32, km_seed, pixel_count as u32) as usize * 4;
        centroids.push([pixels[pi], pixels[pi + 1], pixels[pi + 2]]);
    }
    // Subsample for speed
    let step = (pixel_count / 8192).max(1);
    let samples: Vec<[f32; 3]> = (0..pixel_count)
        .step_by(step)
        .map(|i| {
            let idx = i * 4;
            [pixels[idx], pixels[idx + 1], pixels[idx + 2]]
        })
        .collect();
    for _ in 0..max_iterations {
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0u32; k];
        for s in &samples {
            let mut best_c = 0;
            let mut best_d = f32::MAX;
            for (ci, c) in centroids.iter().enumerate() {
                let d = (s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2) + (s[2] - c[2]).powi(2);
                if d < best_d {
                    best_d = d;
                    best_c = ci;
                }
            }
            sums[best_c][0] += s[0] as f64;
            sums[best_c][1] += s[1] as f64;
            sums[best_c][2] += s[2] as f64;
            counts[best_c] += 1;
        }
        let mut converged = true;
        for (ci, c) in centroids.iter_mut().enumerate() {
            if counts[ci] > 0 {
                let n = counts[ci] as f64;
                let new = [
                    (sums[ci][0] / n) as f32,
                    (sums[ci][1] / n) as f32,
                    (sums[ci][2] / n) as f32,
                ];
                if (new[0] - c[0]).abs() > 1e-5
                    || (new[1] - c[1]).abs() > 1e-5
                    || (new[2] - c[2]).abs() > 1e-5
                {
                    converged = false;
                }
                *c = new;
            }
        }
        if converged {
            break;
        }
    }
    centroids.iter().map(|c| (c[0], c[1], c[2])).collect()
}

pub(crate) fn bayer_matrix(size: u32) -> Vec<f32> {
    let n = size as usize;
    let mut m = vec![0.0f32; n * n];
    // Recursive Bayer construction
    // Base: 2x2
    if n == 2 {
        m[0] = 0.0 / 4.0;
        m[1] = 2.0 / 4.0;
        m[2] = 3.0 / 4.0;
        m[3] = 1.0 / 4.0;
        return m;
    }
    let half = n / 2;
    let sub = bayer_matrix(half as u32);
    let nn = (n * n) as f32;
    for y in 0..n {
        for x in 0..n {
            let sy = y % half;
            let sx = x % half;
            let base = sub[sy * half + sx] * (half * half) as f32;
            let quadrant = match (y / half, x / half) {
                (0, 0) => 0.0,
                (0, 1) => 2.0,
                (1, 0) => 3.0,
                (1, 1) => 1.0,
                _ => 0.0,
            };
            m[y * n + x] = (4.0 * base + quadrant + 0.5) / nn;
        }
    }
    m
}

// ─── Gradient helpers ─────────────────────────────────────────────────────

pub(crate) fn build_gradient_lut(stops: &[(f32, f32, f32, f32)]) -> Vec<f32> {
    let mut lut = vec![0.0f32; 256 * 3];
    for i in 0..256 {
        let t = i as f32 / 255.0;
        let (r, g, b) = interpolate_gradient(stops, t);
        lut[i * 3] = r;
        lut[i * 3 + 1] = g;
        lut[i * 3 + 2] = b;
    }
    lut
}

pub(crate) fn interpolate_gradient(stops: &[(f32, f32, f32, f32)], t: f32) -> (f32, f32, f32) {
    if stops.len() == 1 {
        return (stops[0].1, stops[0].2, stops[0].3);
    }
    if t <= stops[0].0 {
        return (stops[0].1, stops[0].2, stops[0].3);
    }
    let last = stops.len() - 1;
    if t >= stops[last].0 {
        return (stops[last].1, stops[last].2, stops[last].3);
    }
    for i in 0..last {
        if t >= stops[i].0 && t <= stops[i + 1].0 {
            let range = stops[i + 1].0 - stops[i].0;
            let frac = if range > 1e-7 {
                (t - stops[i].0) / range
            } else {
                0.0
            };
            return (
                stops[i].1 + frac * (stops[i + 1].1 - stops[i].1),
                stops[i].2 + frac * (stops[i + 1].2 - stops[i].2),
                stops[i].3 + frac * (stops[i + 1].3 - stops[i].3),
            );
        }
    }
    (stops[last].1, stops[last].2, stops[last].3)
}

// ─── Gaussian blur helper (used by LabSharpen) ───────────────────────────

/// Simple separable Gaussian blur for a single channel.
pub(crate) fn gaussian_blur_1d(data: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
    let kernel_radius = (radius * 3.0).ceil() as usize;
    if kernel_radius == 0 {
        return data.to_vec();
    }
    let sigma = radius;
    let kernel_size = kernel_radius * 2 + 1;
    let mut kernel = vec![0.0f32; kernel_size];
    let mut sum = 0.0f32;
    for (i, kv) in kernel.iter_mut().enumerate() {
        let x = i as f32 - kernel_radius as f32;
        let v = (-x * x / (2.0 * sigma * sigma)).exp();
        *kv = v;
        sum += v;
    }
    for v in &mut kernel {
        *v /= sum;
    }
    // Horizontal pass
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sx = (x as i32 + k as i32 - kernel_radius as i32)
                    .max(0)
                    .min(w as i32 - 1) as usize;
                acc += data[y * w + sx] * kv;
            }
            temp[y * w + x] = acc;
        }
    }
    // Vertical pass
    let mut result = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let sy = (y as i32 + k as i32 - kernel_radius as i32)
                    .max(0)
                    .min(h as i32 - 1) as usize;
                acc += temp[sy * w + x] * kv;
            }
            result[y * w + x] = acc;
        }
    }
    result
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filters::helpers::{hsl_to_rgb, rgb_to_hsl};
    use crate::ops::Filter;

    // ── Color space roundtrip tests ──

    #[test]
    fn hsl_roundtrip() {
        let colors = [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.5, 0.5),
            (0.2, 0.6, 0.8),
        ];
        for (r, g, b) in colors {
            let (h, s, l) = rgb_to_hsl(r, g, b);
            let (rr, rg, rb) = hsl_to_rgb(h, s, l);
            assert!(
                (r - rr).abs() < 0.01 && (g - rg).abs() < 0.01 && (b - rb).abs() < 0.01,
                "HSL roundtrip failed for ({r}, {g}, {b}): got ({rr}, {rg}, {rb})"
            );
        }
    }

    #[test]
    fn lab_roundtrip() {
        let colors = [
            (0.5, 0.5, 0.5),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.2, 0.6, 0.8),
        ];
        for (r, g, b) in colors {
            let (l, a, bi) = rgb_to_lab(r, g, b);
            let (rr, rg, rb) = lab_to_rgb(l, a, bi);
            assert!(
                (r - rr).abs() < 0.02 && (g - rg).abs() < 0.02 && (b - rb).abs() < 0.02,
                "Lab roundtrip failed for ({r}, {g}, {b}): got ({rr:.4}, {rg:.4}, {rb:.4})"
            );
        }
    }

    #[test]
    fn alpha_preserved_all_filters() {
        let input = vec![0.3, 0.5, 0.7, 0.42]; // weird alpha
        let filters: Vec<Box<dyn Filter>> = vec![
            Box::new(HueRotate { degrees: 45.0 }),
            Box::new(Saturate { factor: 0.5 }),
            Box::new(ChannelMixer {
                matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            }),
            Box::new(Vibrance { amount: 50.0 }),
            Box::new(Sepia { intensity: 0.5 }),
            Box::new(Colorize {
                target_r: 1.0,
                target_g: 0.8,
                target_b: 0.2,
                amount: 0.5,
            }),
            Box::new(Modulate {
                brightness: 1.0,
                saturation: 1.0,
                hue: 0.0,
            }),
            Box::new(PhotoFilter {
                color_r: 1.0,
                color_g: 0.0,
                color_b: 0.0,
                density: 0.5,
                preserve_luminosity: true,
            }),
            Box::new(WhiteBalanceTemperature {
                temperature: 7000.0,
                tint: 0.0,
            }),
            Box::new(LabAdjust {
                a_offset: 10.0,
                b_offset: -5.0,
            }),
            Box::new(Invert),
        ];
        for (i, f) in filters.iter().enumerate() {
            let out = f.compute(&input, 1, 1).unwrap();
            assert_eq!(out[3], 0.42, "Filter {i} should preserve alpha");
        }
    }

    // Use Invert from adjustment module for the alpha test
    use crate::filters::adjustment::Invert;

    #[test]
    fn bayer_matrix_2x2() {
        let m = bayer_matrix(2);
        assert_eq!(m.len(), 4);
        // Values should sum to ~2.0 (each normalized to [0,1))
        let sum: f32 = m.iter().sum();
        assert!((sum - 1.5).abs() < 0.01, "Bayer 2x2 sum: {sum}");
    }

    #[test]
    fn bayer_matrix_4x4() {
        let m = bayer_matrix(4);
        assert_eq!(m.len(), 16);
        // All values should be in [0, 1)
        for v in &m {
            assert!(*v >= 0.0 && *v < 1.0, "Bayer value out of range: {v}");
        }
    }
}
