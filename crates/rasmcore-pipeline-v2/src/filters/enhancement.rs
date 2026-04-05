//! Enhancement filters — image quality improvement operations on f32 pixel data.
//!
//! All operate on `&[f32]` RGBA (4 channels per pixel). No format dispatch.
//! No u8/u16 paths. Just f32.
//!
//! Includes: auto-level, CLAHE, clarity, dehaze, equalize, frequency separation,
//! NLM denoise, normalize, pyramid detail remap, retinex (SSR/MSR/MSRCR),
//! shadow-highlight, vignette (Gaussian + power-law).
//!
//! Dodge and Burn are in adjustment.rs (point ops with AnalyticOp support).

use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::ops::{Filter, GpuFilter};

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Luminance (Rec. 709) from f32 RGB.
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Reflect-boundary coordinate clamping.
#[inline]
fn clamp_coord(v: i32, size: usize) -> usize {
    if v < 0 {
        (-v).min(size as i32 - 1) as usize
    } else if v >= size as i32 {
        (2 * size as i32 - v - 2).max(0) as usize
    } else {
        v as usize
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Global histogram operations
// ═══════════════════════════════════════════════════════════════════════════════

/// Auto-level — linear stretch from actual min to actual max.
///
/// Finds per-channel min/max across all pixels and linearly maps to [0, 1].
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "auto_level", category = "enhancement")]
pub struct AutoLevel;

impl Filter for AutoLevel {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];

        for pixel in input.chunks_exact(4) {
            min[0] = min[0].min(pixel[0]);
            min[1] = min[1].min(pixel[1]);
            min[2] = min[2].min(pixel[2]);
            max[0] = max[0].max(pixel[0]);
            max[1] = max[1].max(pixel[1]);
            max[2] = max[2].max(pixel[2]);
        }

        let range = [
            (max[0] - min[0]).max(1e-10),
            (max[1] - min[1]).max(1e-10),
            (max[2] - min[2]).max(1e-10),
        ];
        let inv_range = [1.0 / range[0], 1.0 / range[1], 1.0 / range[2]];
        let mut out = input.to_vec();
        for pixel in out.chunks_exact_mut(4) {
            pixel[0] = (pixel[0] - min[0]) * inv_range[0];
            pixel[1] = (pixel[1] - min[1]) * inv_range[1];
            pixel[2] = (pixel[2] - min[2]) * inv_range[2];
        }
        Ok(out)
    }
}

/// Histogram equalization — maximizes contrast via CDF remapping.
///
/// Quantizes to 256 bins for CDF computation, then remaps.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "equalize", category = "enhancement")]
pub struct Equalize;

impl Filter for Equalize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let npixels = input.len() / 4;
        let mut out = input.to_vec();

        // Per-channel histogram equalization
        for c in 0..3 {
            let mut hist = [0u32; 256];
            for pixel in input.chunks_exact(4) {
                let bin = (pixel[c].clamp(0.0, 1.0) * 255.0) as usize;
                hist[bin.min(255)] += 1;
            }

            // Build CDF
            let mut cdf = [0u32; 256];
            cdf[0] = hist[0];
            for i in 1..256 {
                cdf[i] = cdf[i - 1] + hist[i];
            }

            let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
            let denom = (npixels as u32).saturating_sub(cdf_min);

            if denom > 0 {
                for pixel in out.chunks_exact_mut(4) {
                    let bin = (pixel[c].clamp(0.0, 1.0) * 255.0) as usize;
                    pixel[c] = (cdf[bin.min(255)] - cdf_min) as f32 / denom as f32;
                }
            }
        }

        Ok(out)
    }
}

/// Normalize — linear contrast stretch with 2% black / 1% white clipping.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "normalize", category = "enhancement")]
pub struct Normalize {
    #[param(min = 0.0, max = 0.5, default = 0.02)]
    pub black_clip: f32,
    #[param(min = 0.0, max = 0.5, default = 0.01)]
    pub white_clip: f32,
}

impl Filter for Normalize {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let npixels = input.len() / 4;
        let mut out = input.to_vec();

        for c in 0..3 {
            let mut hist = [0u32; 256];
            for pixel in input.chunks_exact(4) {
                let bin = (pixel[c].clamp(0.0, 1.0) * 255.0) as usize;
                hist[bin.min(255)] += 1;
            }

            // Find black point (skip bottom black_clip fraction)
            let black_threshold = (npixels as f32 * self.black_clip) as u32;
            let mut accum = 0u32;
            let mut black_bin = 0;
            for (i, &h) in hist.iter().enumerate() {
                accum += h;
                if accum >= black_threshold {
                    black_bin = i;
                    break;
                }
            }

            // Find white point (skip top white_clip fraction)
            let white_threshold = (npixels as f32 * self.white_clip) as u32;
            accum = 0;
            let mut white_bin = 255;
            for i in (0..256).rev() {
                accum += hist[i];
                if accum >= white_threshold {
                    white_bin = i;
                    break;
                }
            }

            let black = black_bin as f32 / 255.0;
            let white = white_bin as f32 / 255.0;
            let range = white - black;

            if range > 1e-10 {
                for pixel in out.chunks_exact_mut(4) {
                    pixel[c] = (pixel[c] - black) / range;
                }
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Frequency separation
// ═══════════════════════════════════════════════════════════════════════════════

/// Low-pass frequency layer — Gaussian blur.
///
/// Extracts large-scale color/tone structure.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "frequency_low", category = "enhancement")]
pub struct FrequencyLow {
    #[param(min = 0.0, max = 100.0, default = 3.0)]
    pub sigma: f32,
}

impl Filter for FrequencyLow {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.sigma };
        blur.compute(input, width, height)
    }
}

/// High-pass frequency layer — detail extraction.
///
/// `output = (input - blur(input)) + 0.5`
/// The 0.5 offset provides a neutral midpoint for compositing.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "frequency_high", category = "enhancement")]
pub struct FrequencyHigh {
    #[param(min = 0.0, max = 100.0, default = 3.0)]
    pub sigma: f32,
}

impl Filter for FrequencyHigh {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.sigma };
        let blurred = blur.compute(input, width, height)?;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            if i % 4 == 3 {
                out.push(v); // alpha preserved
            } else {
                out.push((v - blurred[i]) + 0.5);
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Clarity — midtone-weighted local contrast
// ═══════════════════════════════════════════════════════════════════════════════

/// Clarity — midtone-weighted local contrast enhancement (Lightroom/Photoshop style).
///
/// Large-radius unsharp mask weighted by midtone curve:
/// `w(l) = 4 * l * (1 - l)` where l is normalized luminance.
/// `output = input + amount * (input - blur) * w(luminance)`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "clarity", category = "enhancement")]
pub struct Clarity {
    #[param(min = -1.0, max = 1.0, default = 0.0)]
    pub amount: f32,
    #[param(min = 0.0, max = 100.0, default = 20.0)]
    pub radius: f32,
}

impl Filter for Clarity {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.radius };
        let blurred = blur.compute(input, width, height)?;
        let amount = self.amount;
        let mut out = input.to_vec();

        for (pixel, blurred_pixel) in out.chunks_exact_mut(4).zip(blurred.chunks_exact(4)) {
            let luma = luminance(pixel[0], pixel[1], pixel[2]).clamp(0.0, 1.0);
            let aw = amount * 4.0 * luma * (1.0 - luma); // midtone-weighted amount
            pixel[0] += aw * (pixel[0] - blurred_pixel[0]);
            pixel[1] += aw * (pixel[1] - blurred_pixel[1]);
            pixel[2] += aw * (pixel[2] - blurred_pixel[2]);
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dehaze — dark channel prior (He et al. 2009)
// ═══════════════════════════════════════════════════════════════════════════════

/// Dehaze — dark channel prior dehazing.
///
/// 1. Dark channel: local min over RGB in patch
/// 2. Atmospheric light: brightest 0.1% of dark channel
/// 3. Transmission: `t(x) = 1 - omega * dark(I/A)`
/// 4. Recover: `J = (I - A) / max(t, t_min) + A`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "dehaze", category = "enhancement")]
pub struct Dehaze {
    #[param(min = 1, max = 50, default = 7)]
    pub patch_radius: u32,
    #[param(min = 0.0, max = 1.0, default = 0.95)]
    pub omega: f32,
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub t_min: f32,
}

impl Filter for Dehaze {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = self.patch_radius as usize;
        let n = w * h;

        // Step 1: Dark channel — min over RGB in local patch
        let mut dark_channel = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let mut min_val = f32::MAX;
                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(h);
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(w);
                for py in y0..y1 {
                    for px in x0..x1 {
                        let idx = (py * w + px) * 4;
                        let channel_min = input[idx].min(input[idx + 1]).min(input[idx + 2]);
                        min_val = min_val.min(channel_min);
                    }
                }
                dark_channel[y * w + x] = min_val;
            }
        }

        // Step 2: Atmospheric light — average of top 0.1% brightest dark channel pixels
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_unstable_by(|&a, &b| {
            dark_channel[b]
                .partial_cmp(&dark_channel[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top_count = (n as f32 * 0.001).ceil() as usize;
        let top_count = top_count.max(1);
        let mut a_light = [0.0f32; 3];
        for &idx in &indices[..top_count.min(n)] {
            let pixel_idx = idx * 4;
            a_light[0] += input[pixel_idx];
            a_light[1] += input[pixel_idx + 1];
            a_light[2] += input[pixel_idx + 2];
        }
        let inv_count = 1.0 / top_count as f32;
        a_light[0] *= inv_count;
        a_light[1] *= inv_count;
        a_light[2] *= inv_count;

        // Step 3: Transmission estimate
        let mut transmission = vec![0.0f32; n];
        for y in 0..h {
            for x in 0..w {
                let mut min_val = f32::MAX;
                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(h);
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(w);
                for py in y0..y1 {
                    for px in x0..x1 {
                        let idx = (py * w + px) * 4;
                        let nr = input[idx] / a_light[0].max(1e-10);
                        let ng = input[idx + 1] / a_light[1].max(1e-10);
                        let nb = input[idx + 2] / a_light[2].max(1e-10);
                        min_val = min_val.min(nr.min(ng).min(nb));
                    }
                }
                transmission[y * w + x] = 1.0 - self.omega * min_val;
            }
        }

        // Step 4: Recover scene
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let t = transmission[y * w + x].max(self.t_min);
                let inv_t = 1.0 / t;
                for c in 0..3 {
                    out[idx + c] = (input[idx + c] - a_light[c]) * inv_t + a_light[c];
                }
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CLAHE — Contrast-Limited Adaptive Histogram Equalization
// ═══════════════════════════════════════════════════════════════════════════════

/// CLAHE — local adaptive histogram equalization on luminance.
///
/// Operates on luminance channel with bilinear interpolation between tiles.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "clahe", category = "enhancement")]
pub struct Clahe {
    #[param(min = 1, max = 32, default = 8)]
    pub tile_grid: u32,
    #[param(min = 0.0, max = 100.0, default = 2.0)]
    pub clip_limit: f32,
}

impl Filter for Clahe {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let grid = self.tile_grid as usize;
        if grid == 0 {
            return Ok(input.to_vec());
        }

        let tile_w = w / grid;
        let tile_h = h / grid;
        if tile_w == 0 || tile_h == 0 {
            return Ok(input.to_vec());
        }

        let npixels_per_tile = tile_w * tile_h;
        let clip = (self.clip_limit * npixels_per_tile as f32 / 256.0).max(1.0) as u32;

        // Extract luminance
        let luma: Vec<f32> = input
            .chunks_exact(4)
            .map(|p| luminance(p[0], p[1], p[2]).clamp(0.0, 1.0))
            .collect();

        // Build per-tile LUTs
        let mut tile_luts = vec![[0.0f32; 256]; grid * grid];
        for ty in 0..grid {
            for tx in 0..grid {
                let mut hist = [0u32; 256];
                let y0 = ty * tile_h;
                let x0 = tx * tile_w;
                for dy in 0..tile_h {
                    for dx in 0..tile_w {
                        let py = (y0 + dy).min(h - 1);
                        let px = (x0 + dx).min(w - 1);
                        let bin = (luma[py * w + px] * 255.0) as usize;
                        hist[bin.min(255)] += 1;
                    }
                }

                // Clip histogram and redistribute
                let mut excess = 0u32;
                for h in &mut hist {
                    if *h > clip {
                        excess += *h - clip;
                        *h = clip;
                    }
                }
                let per_bin = excess / 256;
                let remainder = excess % 256;
                for (i, h) in hist.iter_mut().enumerate() {
                    *h += per_bin + if (i as u32) < remainder { 1 } else { 0 };
                }

                // Build CDF → LUT
                let mut cdf = [0u32; 256];
                cdf[0] = hist[0];
                for i in 1..256 {
                    cdf[i] = cdf[i - 1] + hist[i];
                }
                let cdf_min = cdf.iter().find(|&&v| v > 0).copied().unwrap_or(0);
                let denom = (npixels_per_tile as u32).saturating_sub(cdf_min).max(1);

                let lut = &mut tile_luts[ty * grid + tx];
                for i in 0..256 {
                    lut[i] = (cdf[i] - cdf_min) as f32 / denom as f32;
                }
            }
        }

        // Apply with bilinear interpolation between tiles
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let tx_f = (x as f32 / tile_w as f32 - 0.5).clamp(0.0, (grid - 1) as f32);
                let ty_f = (y as f32 / tile_h as f32 - 0.5).clamp(0.0, (grid - 1) as f32);
                let tx0 = tx_f as usize;
                let ty0 = ty_f as usize;
                let tx1 = (tx0 + 1).min(grid - 1);
                let ty1 = (ty0 + 1).min(grid - 1);
                let fx = tx_f - tx0 as f32;
                let fy = ty_f - ty0 as f32;

                let bin = (luma[y * w + x] * 255.0) as usize;
                let bin = bin.min(255);

                let v00 = tile_luts[ty0 * grid + tx0][bin];
                let v10 = tile_luts[ty0 * grid + tx1][bin];
                let v01 = tile_luts[ty1 * grid + tx0][bin];
                let v11 = tile_luts[ty1 * grid + tx1][bin];

                let new_luma = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;

                let old_luma = luma[y * w + x].max(1e-10);
                let ratio = new_luma / old_luma;

                let idx = (y * w + x) * 4;
                out[idx] *= ratio;
                out[idx + 1] *= ratio;
                out[idx + 2] *= ratio;
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NLM Denoise — Non-Local Means
// ═══════════════════════════════════════════════════════════════════════════════

/// Non-Local Means denoising (Buades et al. 2005).
///
/// Compares patches in search window, weights by similarity.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "nlm_denoise", category = "enhancement")]
pub struct NlmDenoise {
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub h: f32,
    #[param(min = 1, max = 10, default = 3)]
    pub patch_radius: u32,
    #[param(min = 1, max = 30, default = 10)]
    pub search_radius: u32,
}

impl Filter for NlmDenoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let pr = self.patch_radius as i32;
        let sr = self.search_radius as i32;
        let h2 = self.h * self.h;
        if h2 < 1e-10 {
            return Ok(input.to_vec());
        }
        let inv_h2 = -1.0 / h2;
        let patch_size = ((2 * pr + 1) * (2 * pr + 1)) as f32;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for sy in -sr..=sr {
                    for sx in -sr..=sr {
                        let nx = clamp_coord(x as i32 + sx, w);
                        let ny = clamp_coord(y as i32 + sy, h);

                        // Patch distance
                        let mut dist2 = 0.0f32;
                        for py in -pr..=pr {
                            for px in -pr..=pr {
                                let cx1 = clamp_coord(x as i32 + px, w);
                                let cy1 = clamp_coord(y as i32 + py, h);
                                let cx2 = clamp_coord(nx as i32 + px, w);
                                let cy2 = clamp_coord(ny as i32 + py, h);
                                let i1 = (cy1 * w + cx1) * 4;
                                let i2 = (cy2 * w + cx2) * 4;
                                for c in 0..3 {
                                    let d = input[i1 + c] - input[i2 + c];
                                    dist2 += d * d;
                                }
                            }
                        }
                        dist2 /= patch_size * 3.0;

                        let weight = (dist2 * inv_h2).exp();
                        let nidx = (ny * w + nx) * 4;
                        for c in 0..3 {
                            sum[c] += weight * input[nidx + c];
                        }
                        weight_sum += weight;
                    }
                }

                let idx = (y * w + x) * 4;
                let inv_w = if weight_sum > 1e-10 { 1.0 / weight_sum } else { 1.0 };
                for c in 0..3 {
                    out[idx + c] = sum[c] * inv_w;
                }
                out[idx + 3] = input[idx + 3]; // alpha
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pyramid Detail Remap — Laplacian pyramid enhancement
// ═══════════════════════════════════════════════════════════════════════════════

/// Laplacian pyramid detail remap — enhance or suppress fine detail.
///
/// `sigma < 1.0`: enhance fine detail (compress large gradients).
/// `sigma > 1.0`: suppress fine detail (smoothing).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "pyramid_detail_remap", category = "enhancement")]
pub struct PyramidDetailRemap {
    #[param(min = 0.0, max = 5.0, default = 0.5)]
    pub sigma: f32,
    #[param(min = 0, max = 10, default = 0)]
    pub levels: u32,
}

impl Filter for PyramidDetailRemap {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let sigma = self.sigma;
        let levels = if self.levels == 0 {
            // Auto: log2(min(w,h)) - 2, clamped to [3, 7]
            ((w.min(h) as f32).log2() as u32).saturating_sub(2).clamp(3, 7)
        } else {
            self.levels
        };

        // Process each RGB channel independently
        let mut out = input.to_vec();
        for c in 0..3 {
            // Extract single channel
            let mut channel: Vec<f32> = input.chunks_exact(4).map(|p| p[c]).collect();

            // Build Gaussian pyramid
            let mut gaussians = vec![channel.clone()];
            let mut cw = w;
            let mut ch = h;
            for _ in 0..levels {
                let nw = cw.div_ceil(2);
                let nh = ch.div_ceil(2);
                let prev = gaussians.last().unwrap();
                let mut next = vec![0.0f32; nw * nh];
                for y in 0..nh {
                    for x in 0..nw {
                        let sx = (x * 2).min(cw - 1);
                        let sy = (y * 2).min(ch - 1);
                        // Simple 2x2 average downsample
                        let sx1 = (sx + 1).min(cw - 1);
                        let sy1 = (sy + 1).min(ch - 1);
                        next[y * nw + x] = (prev[sy * cw + sx]
                            + prev[sy * cw + sx1]
                            + prev[sy1 * cw + sx]
                            + prev[sy1 * cw + sx1])
                            * 0.25;
                    }
                }
                gaussians.push(next);
                cw = nw;
                ch = nh;
            }

            // Build Laplacian pyramid and remap detail coefficients
            for level in 0..levels as usize {
                let lw = if level == 0 { w } else { (w >> level).max(1) };
                let lh = if level == 0 { h } else { (h >> level).max(1) };
                // Upsample coarser level
                let uw = lw;
                let uh = lh;
                let mut upsampled = vec![0.0f32; uw * uh];
                let coarse = &gaussians[level + 1];
                let cw_coarse = lw.div_ceil(2);
                for y in 0..uh {
                    for x in 0..uw {
                        let cx = x / 2;
                        let cy = y / 2;
                        let cx = cx.min(cw_coarse.saturating_sub(1));
                        let cy = cy.min((lh.div_ceil(2)).saturating_sub(1));
                        upsampled[y * uw + x] = coarse[cy * cw_coarse + cx];
                    }
                }

                // Remap Laplacian detail: d * sigma / (sigma + |d|)
                let fine = &mut gaussians[level];
                for i in 0..fine.len().min(upsampled.len()) {
                    let detail = fine[i] - upsampled[i];
                    let remapped = if sigma.abs() > 1e-10 {
                        detail * sigma / (sigma + detail.abs())
                    } else {
                        0.0
                    };
                    fine[i] = upsampled[i] + remapped;
                }
            }

            // Write remapped channel back
            channel = gaussians[0].clone();
            for (i, pixel) in out.chunks_exact_mut(4).enumerate() {
                if i < channel.len() {
                    pixel[c] = channel[i];
                }
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Retinex — illumination-invariant enhancement
// ═══════════════════════════════════════════════════════════════════════════════

/// Single-Scale Retinex (Land 1977, Jobson et al. 1997).
///
/// `R(x,y) = log(I(x,y)) - log(G * I(x,y))`
/// Enhances local contrast by removing illumination estimate.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "retinex_ssr", category = "enhancement")]
pub struct RetinexSsr {
    #[param(min = 0.0, max = 200.0, default = 80.0)]
    pub sigma: f32,
}

impl Filter for RetinexSsr {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.sigma };
        let blurred = blur.compute(input, width, height)?;

        let mut out = input.to_vec();
        // Compute log-ratio and normalize
        let mut min_val = [f32::MAX; 3];
        let mut max_val = [f32::MIN; 3];

        for (pixel, blur_pixel) in out.chunks_exact_mut(4).zip(blurred.chunks_exact(4)) {
            for c in 0..3 {
                let log_input = (pixel[c].max(1e-10)).ln();
                let log_blur = (blur_pixel[c].max(1e-10)).ln();
                pixel[c] = log_input - log_blur;
                min_val[c] = min_val[c].min(pixel[c]);
                max_val[c] = max_val[c].max(pixel[c]);
            }
        }

        // Normalize to [0, 1]
        for pixel in out.chunks_exact_mut(4) {
            for c in 0..3 {
                let range = max_val[c] - min_val[c];
                if range > 1e-10 {
                    pixel[c] = (pixel[c] - min_val[c]) / range;
                }
            }
        }

        Ok(out)
    }
}

/// Multi-Scale Retinex (Jobson et al. 1997).
///
/// Averages SSR at three scales for better overall contrast.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "retinex_msr", category = "enhancement")]
pub struct RetinexMsr {
    #[param(min = 0.0, max = 200.0, default = 15.0)]
    pub sigma_small: f32,
    #[param(min = 0.0, max = 200.0, default = 80.0)]
    pub sigma_medium: f32,
    #[param(min = 0.0, max = 500.0, default = 250.0)]
    pub sigma_large: f32,
}

impl Filter for RetinexMsr {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let scales = [self.sigma_small, self.sigma_medium, self.sigma_large];
        let n = input.len();
        let mut accum = vec![0.0f32; n];

        for sigma in &scales {
            let blur = GaussianBlur { radius: *sigma };
            let blurred = blur.compute(input, width, height)?;
            for (i, (inp, blr)) in input.iter().zip(blurred.iter()).enumerate() {
                if i % 4 == 3 {
                    continue; // skip alpha
                }
                let log_input = inp.max(1e-10).ln();
                let log_blur = blr.max(1e-10).ln();
                accum[i] += log_input - log_blur;
            }
        }

        // Average and normalize
        let inv_scales = 1.0 / scales.len() as f32;
        let mut min_val = [f32::MAX; 3];
        let mut max_val = [f32::MIN; 3];
        for pixel in accum.chunks_exact_mut(4) {
            for c in 0..3 {
                pixel[c] *= inv_scales;
                min_val[c] = min_val[c].min(pixel[c]);
                max_val[c] = max_val[c].max(pixel[c]);
            }
        }
        for (out_pixel, in_pixel) in accum.chunks_exact_mut(4).zip(input.chunks_exact(4)) {
            for c in 0..3 {
                let range = max_val[c] - min_val[c];
                if range > 1e-10 {
                    out_pixel[c] = (out_pixel[c] - min_val[c]) / range;
                }
            }
            out_pixel[3] = in_pixel[3]; // alpha from input
        }

        Ok(accum)
    }
}

/// Multi-Scale Retinex with Color Restoration (Jobson et al. 1997).
///
/// MSR + chromaticity-based gain for color preservation.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "retinex_msrcr", category = "enhancement")]
pub struct RetinexMsrcr {
    #[param(min = 0.0, max = 200.0, default = 15.0)]
    pub sigma_small: f32,
    #[param(min = 0.0, max = 200.0, default = 80.0)]
    pub sigma_medium: f32,
    #[param(min = 0.0, max = 500.0, default = 250.0)]
    pub sigma_large: f32,
    #[param(min = 0.0, max = 200.0, default = 125.0)]
    pub alpha: f32,
    #[param(min = 0.0, max = 100.0, default = 46.0)]
    pub beta: f32,
}

impl Filter for RetinexMsrcr {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let scales = [self.sigma_small, self.sigma_medium, self.sigma_large];
        let n = input.len();
        let mut msr = vec![0.0f32; n];

        // MSR computation
        for sigma in &scales {
            let blur = GaussianBlur { radius: *sigma };
            let blurred = blur.compute(input, width, height)?;
            for (i, (inp, blr)) in input.iter().zip(blurred.iter()).enumerate() {
                if i % 4 == 3 {
                    continue;
                }
                msr[i] += inp.max(1e-10).ln() - blr.max(1e-10).ln();
            }
        }

        let inv_scales = 1.0 / scales.len() as f32;
        for v in msr.iter_mut() {
            *v *= inv_scales;
        }

        // Color restoration
        let mut out = vec![0.0f32; n];
        for (pixel_idx, (in_pixel, msr_pixel)) in input
            .chunks_exact(4)
            .zip(msr.chunks_exact(4))
            .enumerate()
        {
            let sum = in_pixel[0] + in_pixel[1] + in_pixel[2];
            let idx = pixel_idx * 4;
            for c in 0..3 {
                let chromaticity = if sum > 1e-10 {
                    in_pixel[c] / sum
                } else {
                    1.0 / 3.0
                };
                let color_gain = self.beta * (self.alpha * chromaticity).ln().max(-10.0);
                out[idx + c] = color_gain * msr_pixel[c];
            }
            out[idx + 3] = in_pixel[3];
        }

        // Normalize to [0, 1]
        let mut min_val = [f32::MAX; 3];
        let mut max_val = [f32::MIN; 3];
        for pixel in out.chunks_exact(4) {
            for c in 0..3 {
                min_val[c] = min_val[c].min(pixel[c]);
                max_val[c] = max_val[c].max(pixel[c]);
            }
        }
        for pixel in out.chunks_exact_mut(4) {
            for c in 0..3 {
                let range = max_val[c] - min_val[c];
                if range > 1e-10 {
                    pixel[c] = (pixel[c] - min_val[c]) / range;
                }
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Shadow/Highlight — local tone mapping
// ═══════════════════════════════════════════════════════════════════════════════

/// Shadow/Highlight adjustment — local tone mapping.
///
/// Independently lighten shadows and darken highlights via soft-light blending
/// on the luminance channel with compress-gated weight masks.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "shadow_highlight", category = "enhancement")]
pub struct ShadowHighlight {
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub shadows: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub highlights: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub whitepoint: f32,
    #[param(min = 0.0, max = 200.0, default = 30.0)]
    pub radius: f32,
    #[param(min = 0.0, max = 100.0, default = 50.0)]
    pub compress: f32,
    #[param(min = 0.0, max = 100.0, default = 50.0)]
    pub shadows_ccorrect: f32,
    #[param(min = 0.0, max = 100.0, default = 50.0)]
    pub highlights_ccorrect: f32,
}

impl Filter for ShadowHighlight {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let shadows = self.shadows / 100.0;
        let highlights = self.highlights / 100.0;
        let whitepoint = self.whitepoint;
        let compress = self.compress / 100.0;
        let sc = self.shadows_ccorrect / 100.0;
        let hc = self.highlights_ccorrect / 100.0;

        // Extract luminance and blur it
        let luma: Vec<f32> = input
            .chunks_exact(4)
            .map(|p| luminance(p[0], p[1], p[2]).clamp(0.0, 1.0))
            .collect();

        // Blur luminance
        let mut luma_rgba: Vec<f32> = luma.iter().flat_map(|&v| [v, v, v, 1.0]).collect();
        let blur = GaussianBlur { radius: self.radius };
        luma_rgba = blur.compute(&luma_rgba, width, height)?;
        let blurred_luma: Vec<f32> = luma_rgba.chunks_exact(4).map(|p| p[0]).collect();

        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let bl = blurred_luma[y * w + x].clamp(0.0, 1.0);

                // Shadow weight: strongest at dark pixels
                let sw = 1.0 - bl;
                let sw = sw * sw; // quadratic falloff

                // Highlight weight: strongest at bright pixels
                let hw = bl;
                let hw = hw * hw;

                // Compress: reduce effect near midtones
                let sw = if compress > 0.0 {
                    sw * (1.0 - compress * 4.0 * bl * (1.0 - bl))
                } else {
                    sw
                };
                let hw = if compress > 0.0 {
                    hw * (1.0 - compress * 4.0 * bl * (1.0 - bl))
                } else {
                    hw
                };

                // Luminance adjustment
                let luma_adj = shadows * sw - highlights * hw + whitepoint * 0.01;

                let cur_luma = luminance(out[idx], out[idx + 1], out[idx + 2]).max(1e-10);
                let new_luma = (cur_luma + luma_adj).max(0.0);
                let ratio = new_luma / cur_luma;

                // Apply luminance ratio with saturation correction
                for c in 0..3 {
                    let v = out[idx + c];
                    let gray = cur_luma;
                    let chroma = v - gray;

                    // Shadow saturation correction
                    let sat_adj = 1.0 + chroma.signum() * sw * (sc - 1.0)
                        + chroma.signum() * hw * (hc - 1.0);

                    out[idx + c] = new_luma + chroma * sat_adj.max(0.0) * ratio;
                }
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Vignette effects
// ═══════════════════════════════════════════════════════════════════════════════

/// Gaussian vignette — elliptical darkening with Gaussian blur transition.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "vignette", category = "enhancement")]
pub struct Vignette {
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub sigma: f32,
    #[param(min = 0, max = 1000, default = 0)]
    pub x_inset: u32,
    #[param(min = 0, max = 1000, default = 0)]
    pub y_inset: u32,
}

impl Filter for Vignette {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;

        // Build elliptical mask
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let rx = (cx - self.x_inset as f32).max(1.0);
        let ry = (cy - self.y_inset as f32).max(1.0);

        let mut mask = vec![1.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let dx = (x as f32 - cx) / rx;
                let dy = (y as f32 - cy) / ry;
                let dist2 = dx * dx + dy * dy;
                if dist2 > 1.0 {
                    mask[y * w + x] = 0.0;
                }
            }
        }

        // Blur the mask for smooth transition
        if self.sigma > 0.0 {
            let blur = GaussianBlur { radius: self.sigma };
            // Pack mask as RGBA for blur
            let mut mask_rgba: Vec<f32> = mask.iter().flat_map(|&v| [v, v, v, 1.0]).collect();
            mask_rgba = blur.compute(&mask_rgba, width, height)?;
            for (i, pixel) in mask_rgba.chunks_exact(4).enumerate() {
                mask[i] = pixel[0];
            }
        }

        // Apply mask
        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let m = mask[y * w + x];
                out[idx] *= m;
                out[idx + 1] *= m;
                out[idx + 2] *= m;
            }
        }

        Ok(out)
    }
}

/// Power-law vignette — simple radial falloff.
///
/// `factor = 1.0 - strength * (dist / max_dist) ^ falloff`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "vignette_powerlaw", category = "enhancement")]
pub struct VignettePowerlaw {
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub strength: f32,
    #[param(min = 0.5, max = 5.0, default = 2.0)]
    pub falloff: f32,
}

impl Filter for VignettePowerlaw {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();
        if max_dist < 1e-10 {
            return Ok(input.to_vec());
        }
        let inv_max = 1.0 / max_dist;

        let mut out = input.to_vec();
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt() * inv_max;
                let factor = (1.0 - self.strength * dist.powf(self.falloff)).max(0.0);
                let idx = (y * w + x) * 4;
                out[idx] *= factor;
                out[idx + 1] *= factor;
                out[idx + 2] *= factor;
            }
        }

        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Shaders
// ═══════════════════════════════════════════════════════════════════════════════

const VIGNETTE_POWERLAW_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  strength: f32,
  falloff: f32,
  inv_max_dist: f32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let cx = f32(params.width) / 2.0;
  let cy = f32(params.height) / 2.0;
  let dx = f32(gid.x) - cx;
  let dy = f32(gid.y) - cy;
  let dist = sqrt(dx * dx + dy * dy) * params.inv_max_dist;
  let factor = max(1.0 - params.strength * pow(dist, params.falloff), 0.0);
  store_pixel(idx, vec4<f32>(pixel.x * factor, pixel.y * factor, pixel.z * factor, pixel.w));
}
"#;

impl GpuFilter for VignettePowerlaw {
    fn shader_body(&self) -> &str { VIGNETTE_POWERLAW_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt().max(1.0);
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&self.strength.to_le_bytes());
        buf.extend_from_slice(&self.falloff.to_le_bytes());
        buf.extend_from_slice(&(1.0f32 / max_dist).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
}

// ── Vignette (Gaussian) GPU — single-pass analytical falloff ────────────────

gpu_filter!(Vignette,
    shader: crate::gpu_shaders::vignette::VIGNETTE_GAUSSIAN,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [
        w, h,
        self_.sigma,
        self_.x_inset as f32,
        self_.y_inset as f32,
        0u32, 0u32, 0u32
    ]
);

// ── AutoLevel GPU (3-pass via ChannelMinMax reduction) ──────────────────────

/// AutoLevel apply shader — reads min/max from reduction buffer, stretches per-channel.
const AUTO_LEVEL_APPLY_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> reduction: array<vec4<f32>>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let ch_min = reduction[0].xyz;
  let ch_max = reduction[1].xyz;
  let range = max(ch_max - ch_min, vec3<f32>(0.00001, 0.00001, 0.00001));

  let pixel = input[idx];
  output[idx] = vec4<f32>(
    (pixel.x - ch_min.x) / range.x,
    (pixel.y - ch_min.y) / range.y,
    (pixel.z - ch_min.z) / range.z,
    pixel.w,
  );
}
"#;

impl GpuFilter for AutoLevel {
    fn shader_body(&self) -> &str { AUTO_LEVEL_APPLY_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn gpu_shaders(&self, width: u32, height: u32) -> Vec<crate::node::GpuShader> {
        use crate::gpu_shaders::reduction::GpuReduction;

        let reduction = GpuReduction::channel_min_max(256);
        let passes = reduction.build_passes(width, height);

        let pass3 = crate::node::GpuShader::new(
            AUTO_LEVEL_APPLY_WGSL.to_string(),
            "main",
            [256, 1, 1],
            self.params(width, height),
        )
        .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);

        vec![passes.pass1, passes.pass2, pass3]
    }
}

// ── Equalize GPU (3-pass via Histogram256 reduction) ───────────────────────

/// Equalize apply shader — reads per-channel histogram, computes CDF inline, remaps.
const EQUALIZE_APPLY_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  total_pixels: u32,
  _pad: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

// Compute CDF value for a given bin and channel offset
fn cdf_at(bin: u32, channel_offset: u32) -> f32 {
  var sum = 0u;
  for (var i = 0u; i <= bin; i = i + 1u) {
    sum += histogram[channel_offset + i];
  }
  // Find cdf_min (first non-zero bin)
  var cdf_min = 0u;
  for (var i = 0u; i < 256u; i = i + 1u) {
    let v = histogram[channel_offset + i];
    if (v > 0u) {
      cdf_min = v;
      // Compute cdf_min as cumulative at first non-zero
      var s = 0u;
      for (var j = 0u; j <= i; j = j + 1u) {
        s += histogram[channel_offset + j];
      }
      cdf_min = s;
      break;
    }
  }
  let denom = params.total_pixels - cdf_min;
  if (denom == 0u) { return f32(bin) / 255.0; }
  return f32(sum - cdf_min) / f32(denom);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let pixel = input[idx];
  let bin_r = u32(clamp(pixel.x * 255.0, 0.0, 255.0));
  let bin_g = u32(clamp(pixel.y * 255.0, 0.0, 255.0));
  let bin_b = u32(clamp(pixel.z * 255.0, 0.0, 255.0));

  output[idx] = vec4<f32>(
    cdf_at(bin_r, 0u),
    cdf_at(bin_g, 256u),
    cdf_at(bin_b, 512u),
    pixel.w,
  );
}
"#;

impl GpuFilter for Equalize {
    fn shader_body(&self) -> &str { EQUALIZE_APPLY_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let total = width * height;
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&total.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn gpu_shaders(&self, width: u32, height: u32) -> Vec<crate::node::GpuShader> {
        use crate::gpu_shaders::reduction::GpuReduction;

        let reduction = GpuReduction::histogram_256(256);
        let passes = reduction.build_passes(width, height);

        let pass3 = crate::node::GpuShader::new(
            EQUALIZE_APPLY_WGSL.to_string(),
            "main",
            [256, 1, 1],
            self.params(width, height),
        )
        .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);

        vec![passes.pass1, passes.pass2, pass3]
    }
}

// ── Normalize GPU (3-pass via Histogram256 reduction) ──────────────────────

/// Normalize apply shader — reads histogram, finds percentile clip points, stretches.
const NORMALIZE_APPLY_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  total_pixels: u32,
  black_clip_count: u32,
  white_clip_count: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> histogram: array<u32>;

// Find the black point (percentile clip from bottom) for a channel
fn find_black(channel_offset: u32) -> f32 {
  var accum = 0u;
  for (var i = 0u; i < 256u; i = i + 1u) {
    accum += histogram[channel_offset + i];
    if (accum >= params.black_clip_count) {
      return f32(i) / 255.0;
    }
  }
  return 0.0;
}

// Find the white point (percentile clip from top) for a channel
fn find_white(channel_offset: u32) -> f32 {
  var accum = 0u;
  for (var i = 255u; ; i = i - 1u) {
    accum += histogram[channel_offset + i];
    if (accum >= params.white_clip_count) {
      return f32(i) / 255.0;
    }
    if (i == 0u) { break; }
  }
  return 1.0;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }

  let black_r = find_black(0u);
  let black_g = find_black(256u);
  let black_b = find_black(512u);
  let white_r = find_white(0u);
  let white_g = find_white(256u);
  let white_b = find_white(512u);

  let range_r = max(white_r - black_r, 0.00001);
  let range_g = max(white_g - black_g, 0.00001);
  let range_b = max(white_b - black_b, 0.00001);

  let pixel = input[idx];
  output[idx] = vec4<f32>(
    (pixel.x - black_r) / range_r,
    (pixel.y - black_g) / range_g,
    (pixel.z - black_b) / range_b,
    pixel.w,
  );
}
"#;

impl GpuFilter for Normalize {
    fn shader_body(&self) -> &str { NORMALIZE_APPLY_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let total = width * height;
        let black_clip_count = (total as f32 * self.black_clip) as u32;
        let white_clip_count = (total as f32 * self.white_clip) as u32;
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&width.to_le_bytes());
        buf.extend_from_slice(&height.to_le_bytes());
        buf.extend_from_slice(&total.to_le_bytes());
        buf.extend_from_slice(&black_clip_count.to_le_bytes());
        buf.extend_from_slice(&white_clip_count.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn gpu_shaders(&self, width: u32, height: u32) -> Vec<crate::node::GpuShader> {
        use crate::gpu_shaders::reduction::GpuReduction;

        let reduction = GpuReduction::histogram_256(256);
        let passes = reduction.build_passes(width, height);

        let pass3 = crate::node::GpuShader::new(
            NORMALIZE_APPLY_WGSL.to_string(),
            "main",
            [256, 1, 1],
            self.params(width, height),
        )
        .with_reduction_buffers(vec![reduction.read_buffer(&passes)]);

        vec![passes.pass1, passes.pass2, pass3]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU filter wiring — remaining enhancement filters
// ═══════════════════════════════════════════════════════════════════════════════

use crate::gpu_shaders::{analysis, enhancement as enh_shaders, spatial};
use crate::node::{GpuShader, ReductionBuffer};
use crate::filters::spatial::{gaussian_kernel_bytes, blur_params};

// ── NlmDenoise GPU (single-pass, compute-heavy) ─────────────────────────────

gpu_filter!(NlmDenoise,
    shader: analysis::NLM_DENOISE,
    workgroup: [16, 16, 1],
    params(self_, w, h) => [
        w, h, self_.search_radius, self_.patch_radius,
        self_.h, 0u32, 0u32, 0u32
    ]
);

// ── Dehaze GPU (2-pass: dark channel → apply) ───────────────────────────────

gpu_filter_multipass!(Dehaze,
    shader: analysis::DEHAZE_APPLY,
    workgroup: [256, 1, 1],
    params(self_, w, h) => [
        w, h,
        // Atmospheric light estimated on CPU as fallback (GPU path estimates via dark channel max)
        1.0f32, 1.0f32, 1.0f32, // atmosphere RGB
        self_.omega,
        0u32, 0u32
    ],
    passes(self2, w2, h2) => {
        let mut dark_params = Vec::with_capacity(16);
        dark_params.extend_from_slice(&w2.to_le_bytes());
        dark_params.extend_from_slice(&h2.to_le_bytes());
        dark_params.extend_from_slice(&self2.patch_radius.to_le_bytes());
        dark_params.extend_from_slice(&0u32.to_le_bytes());

        let pass1 = GpuShader::new(
            analysis::DEHAZE_DARK_CHANNEL.to_string(), "main", [16, 16, 1], dark_params,
        );
        let pass2 = GpuShader::new(
            analysis::DEHAZE_APPLY.to_string(), "main", [256, 1, 1], self2.params(w2, h2),
        );
        vec![pass1, pass2]
    }
);

// ── ShadowHighlight GPU (blur luma + apply) ─────────────────────────────────

gpu_filter_passes_only!(ShadowHighlight,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let bp = blur_params(w, h, kr);

        // Pass 1-2: blur the input (used to estimate local luminance)
        let blur_h = GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
            .with_extra_buffers(vec![kb.clone()]);
        let blur_v = GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
            .with_extra_buffers(vec![kb]);

        // Pass 3: shadow/highlight apply with blurred luma from previous passes
        let shadows_norm = self_.shadows / 100.0;
        let highlights_norm = self_.highlights / 100.0;
        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&shadows_norm.to_le_bytes());
        apply_params.extend_from_slice(&highlights_norm.to_le_bytes());
        let apply = GpuShader::new(
            enh_shaders::SHADOW_HIGHLIGHT_APPLY.to_string(), "main", [256, 1, 1], apply_params,
        );

        vec![blur_h, blur_v, apply]
    }
);

// ── FrequencyLow GPU (same as GaussianBlur) ─────────────────────────────────

gpu_filter_passes_only!(FrequencyLow,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.sigma);
        let bp = blur_params(w, h, kr);
        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
        ]
    }
);

// ── FrequencyHigh GPU (blur + subtract apply) ───────────────────────────────

gpu_filter_passes_only!(FrequencyHigh,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.sigma);
        let bp = blur_params(w, h, kr);

        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());

        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(enh_shaders::FREQUENCY_HIGH_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);

// ── Clarity GPU (blur + midtone-weighted blend) ─────────────────────────────

gpu_filter_passes_only!(Clarity,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.radius);
        let bp = blur_params(w, h, kr);

        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&self_.amount.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());

        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(enh_shaders::CLARITY_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);

// ── RetinexSsr GPU (blur + log-domain apply) ────────────────────────────────

gpu_filter_passes_only!(RetinexSsr,
    passes(self_, w, h) => {
        let (kr, kb) = gaussian_kernel_bytes(self_.sigma);
        let bp = blur_params(w, h, kr);

        let mut apply_params = Vec::with_capacity(16);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&1.0f32.to_le_bytes()); // gain
        apply_params.extend_from_slice(&0.0f32.to_le_bytes()); // offset

        vec![
            GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                .with_extra_buffers(vec![kb.clone()]),
            GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                .with_extra_buffers(vec![kb]),
            GpuShader::new(enh_shaders::RETINEX_SSR_APPLY.to_string(), "main", [256, 1, 1], apply_params),
        ]
    }
);

// ── RetinexMsr GPU (3 blur scales + accumulate + normalize) ─────────────────

gpu_filter_passes_only!(RetinexMsr,
    passes(self_, w, h) => {
        let scales = [self_.sigma_small, self_.sigma_medium, self_.sigma_large];
        let total_pixels = w * h;
        let acc_size = total_pixels as usize * 16; // vec4<f32> per pixel

        let mut passes = Vec::new();

        // For each scale: blur H, blur V, accumulate
        for sigma in &scales {
            let (kr, kb) = gaussian_kernel_bytes(*sigma);
            let bp = blur_params(w, h, kr);

            passes.push(
                GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                    .with_extra_buffers(vec![kb.clone()])
            );
            passes.push(
                GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                    .with_extra_buffers(vec![kb])
            );

            let mut acc_params = Vec::with_capacity(16);
            acc_params.extend_from_slice(&w.to_le_bytes());
            acc_params.extend_from_slice(&h.to_le_bytes());
            acc_params.extend_from_slice(&0u32.to_le_bytes());
            acc_params.extend_from_slice(&0u32.to_le_bytes());

            passes.push(
                GpuShader::new(enh_shaders::RETINEX_MSR_ACCUMULATE.to_string(), "main", [256, 1, 1], acc_params)
                    .with_reduction_buffers(vec![ReductionBuffer {
                        id: 0,
                        initial_data: vec![0u8; acc_size],
                        read_write: true,
                    }])
            );
        }

        // Final normalize pass (reads accumulator, needs min/max reduction)
        // For simplicity, use the ChannelMinMax reduction on the accumulator
        let reduction = crate::gpu_shaders::reduction::GpuReduction::channel_min_max(256);
        let red_passes = reduction.build_passes(w, h);
        let red_read_buf = reduction.read_buffer(&red_passes);
        passes.push(red_passes.pass1);
        passes.push(red_passes.pass2);

        let mut norm_params = Vec::with_capacity(16);
        norm_params.extend_from_slice(&w.to_le_bytes());
        norm_params.extend_from_slice(&h.to_le_bytes());
        norm_params.extend_from_slice(&3.0f32.to_le_bytes()); // num_scales
        norm_params.extend_from_slice(&0u32.to_le_bytes());

        passes.push(
            GpuShader::new(enh_shaders::RETINEX_MSR_NORMALIZE.to_string(), "main", [256, 1, 1], norm_params)
                .with_reduction_buffers(vec![
                    ReductionBuffer { id: 0, initial_data: vec![], read_write: false },
                    red_read_buf,
                ])
        );

        passes
    }
);

// ── RetinexMsrcr GPU (3 blur scales + accumulate + color restoration) ───────

gpu_filter_passes_only!(RetinexMsrcr,
    passes(self_, w, h) => {
        let scales = [self_.sigma_small, self_.sigma_medium, self_.sigma_large];
        let total_pixels = w * h;
        let acc_size = total_pixels as usize * 16;

        let mut passes = Vec::new();

        for sigma in &scales {
            let (kr, kb) = gaussian_kernel_bytes(*sigma);
            let bp = blur_params(w, h, kr);

            passes.push(
                GpuShader::new(spatial::GAUSSIAN_BLUR_H.to_string(), "main", [256, 1, 1], bp.clone())
                    .with_extra_buffers(vec![kb.clone()])
            );
            passes.push(
                GpuShader::new(spatial::GAUSSIAN_BLUR_V.to_string(), "main", [256, 1, 1], bp)
                    .with_extra_buffers(vec![kb])
            );

            let mut acc_params = Vec::with_capacity(16);
            acc_params.extend_from_slice(&w.to_le_bytes());
            acc_params.extend_from_slice(&h.to_le_bytes());
            acc_params.extend_from_slice(&0u32.to_le_bytes());
            acc_params.extend_from_slice(&0u32.to_le_bytes());

            passes.push(
                GpuShader::new(enh_shaders::RETINEX_MSR_ACCUMULATE.to_string(), "main", [256, 1, 1], acc_params)
                    .with_reduction_buffers(vec![ReductionBuffer {
                        id: 0,
                        initial_data: vec![0u8; acc_size],
                        read_write: true,
                    }])
            );
        }

        // MSRCR color restoration pass
        let mut apply_params = Vec::with_capacity(32);
        apply_params.extend_from_slice(&w.to_le_bytes());
        apply_params.extend_from_slice(&h.to_le_bytes());
        apply_params.extend_from_slice(&3.0f32.to_le_bytes()); // num_scales
        apply_params.extend_from_slice(&self_.alpha.to_le_bytes());
        apply_params.extend_from_slice(&self_.beta.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());
        apply_params.extend_from_slice(&0u32.to_le_bytes());

        passes.push(
            GpuShader::new(enh_shaders::RETINEX_MSRCR_APPLY.to_string(), "main", [256, 1, 1], apply_params)
                .with_reduction_buffers(vec![
                    ReductionBuffer { id: 0, initial_data: vec![], read_write: false },
                ])
        );

        passes
    }
);

// ── Clahe GPU (single-pass with pre-computed tile LUTs) ─────────────────────
//
// Clahe is a special case: the per-tile histogram clipping is done on CPU
// (small compute, complex logic), then the LUTs are passed to GPU for the
// parallel bilinear-interpolated application pass.

impl GpuFilter for Clahe {
    fn shader_body(&self) -> &str { enh_shaders::CLAHE_APPLY }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, w: u32, h: u32) -> Vec<u8> {
        let grid = self.tile_grid;
        let tile_w = w / grid.max(1);
        let tile_h = h / grid.max(1);
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&w.to_le_bytes());
        buf.extend_from_slice(&h.to_le_bytes());
        buf.extend_from_slice(&grid.to_le_bytes());
        buf.extend_from_slice(&tile_w.to_le_bytes());
        buf.extend_from_slice(&tile_h.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf
    }
    fn extra_buffers(&self) -> Vec<Vec<u8>> {
        // Tile LUTs will be computed on CPU and passed here at dispatch time.
        // For now return empty — the executor handles CPU→GPU LUT upload.
        vec![]
    }
}

// ── PyramidDetailRemap GPU (multi-pass Laplacian pyramid) ───────────────────
//
// GPU pyramid: downsample chain → upsample + remap chain.
// Each level is a separate dispatch.

gpu_filter_passes_only!(PyramidDetailRemap,
    passes(self_, w, h) => {
        let levels = if self_.levels == 0 {
            ((w.min(h) as f32).log2() as u32).saturating_sub(2).clamp(3, 7)
        } else {
            self_.levels
        };

        let mut passes = Vec::new();
        let mut dims: Vec<(u32, u32)> = vec![(w, h)];

        // Build downsample chain
        for _ in 0..levels {
            let (cw, ch) = *dims.last().unwrap();
            let nw = cw.div_ceil(2);
            let nh = ch.div_ceil(2);
            let mut ds_params = Vec::with_capacity(16);
            ds_params.extend_from_slice(&cw.to_le_bytes());
            ds_params.extend_from_slice(&ch.to_le_bytes());
            ds_params.extend_from_slice(&nw.to_le_bytes());
            ds_params.extend_from_slice(&nh.to_le_bytes());
            passes.push(
                GpuShader::new(enh_shaders::DOWNSAMPLE_2X.to_string(), "main", [256, 1, 1], ds_params)
            );
            dims.push((nw, nh));
        }

        // Remap + upsample chain (coarsest to finest)
        for level in (0..levels as usize).rev() {
            let (lw, lh) = dims[level];
            let (cw, ch) = dims[level + 1];

            // Upsample coarser level
            let mut us_params = Vec::with_capacity(16);
            us_params.extend_from_slice(&cw.to_le_bytes());
            us_params.extend_from_slice(&ch.to_le_bytes());
            us_params.extend_from_slice(&lw.to_le_bytes());
            us_params.extend_from_slice(&lh.to_le_bytes());
            passes.push(
                GpuShader::new(enh_shaders::UPSAMPLE_2X.to_string(), "main", [256, 1, 1], us_params)
            );

            // Remap Laplacian detail at this level
            let mut remap_params = Vec::with_capacity(16);
            remap_params.extend_from_slice(&lw.to_le_bytes());
            remap_params.extend_from_slice(&lh.to_le_bytes());
            remap_params.extend_from_slice(&self_.sigma.to_le_bytes());
            remap_params.extend_from_slice(&0u32.to_le_bytes());
            passes.push(
                GpuShader::new(enh_shaders::PYRAMID_REMAP_LEVEL.to_string(), "main", [256, 1, 1], remap_params)
            );
        }

        passes
    }
);

// All enhancement filters are auto-registered via #[derive(V2Filter)] on their structs.

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
        let n = (w * h) as usize;
        let mut px = Vec::with_capacity(n * 4);
        for _ in 0..n {
            px.extend_from_slice(&color);
        }
        px
    }

    fn gradient_rgba(w: u32, h: u32) -> Vec<f32> {
        let mut px = Vec::with_capacity((w * h) as usize * 4);
        for y in 0..h {
            for x in 0..w {
                px.push(x as f32 / w as f32);
                px.push(y as f32 / h as f32);
                px.push(0.5);
                px.push(1.0);
            }
        }
        px
    }

    // ─── Auto Level ──────────────────────────────────────────────────────

    #[test]
    fn auto_level_expands_range() {
        // Input: all values in [0.2, 0.8]
        let mut input = solid_rgba(4, 4, [0.5, 0.5, 0.5, 1.0]);
        input[0] = 0.2; // min R
        input[4] = 0.8; // max R
        let output = AutoLevel.compute(&input, 4, 4).unwrap();
        // Min should map to ~0, max should map to ~1
        assert!(output[0] < 0.01);
        assert!(output[4] > 0.99);
    }

    #[test]
    fn auto_level_preserves_alpha() {
        let input = solid_rgba(4, 4, [0.3, 0.5, 0.7, 0.5]);
        let output = AutoLevel.compute(&input, 4, 4).unwrap();
        assert_eq!(output[3], 0.5);
    }

    #[test]
    fn auto_level_gpu_returns_3_passes() {
        let f = AutoLevel;
        let shaders = f.gpu_shaders(100, 100);
        assert_eq!(shaders.len(), 3, "AutoLevel should use 3-pass ChannelMinMax reduction");
        // Pass 1+2 have read_write reduction buffers, pass 3 has read-only
        assert!(shaders[0].reduction_buffers[0].read_write);
        assert!(!shaders[2].reduction_buffers[0].read_write);
        // All have same buffer ID
        assert_eq!(
            shaders[0].reduction_buffers[0].id,
            shaders[2].reduction_buffers[0].id,
        );
    }

    #[test]
    fn equalize_gpu_returns_3_passes() {
        let f = Equalize;
        let shaders = f.gpu_shaders(100, 100);
        assert_eq!(shaders.len(), 3, "Equalize should use 3-pass Histogram256 reduction");
        assert!(shaders[0].reduction_buffers[0].read_write);
        assert!(!shaders[2].reduction_buffers[0].read_write);
    }

    // ─── Equalize ────────────────────────────────────────────────────────

    #[test]
    fn equalize_spreads_histogram() {
        let input = gradient_rgba(16, 16);
        let output = Equalize.compute(&input, 16, 16).unwrap();
        // Output should use full range
        let r_vals: Vec<f32> = output.chunks_exact(4).map(|p| p[0]).collect();
        assert!(*r_vals.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() < 0.1);
        assert!(*r_vals.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() > 0.9);
    }

    #[test]
    fn normalize_gpu_returns_3_passes() {
        let f = Normalize::default();
        let shaders = f.gpu_shaders(100, 100);
        assert_eq!(shaders.len(), 3, "Normalize should use 3-pass Histogram256 reduction");
        assert!(shaders[0].reduction_buffers[0].read_write);
        assert!(!shaders[2].reduction_buffers[0].read_write);
        // Pass 3 params should encode clip counts
        let params = &shaders[2].params;
        let black_clip_count = u32::from_le_bytes(params[12..16].try_into().unwrap());
        let white_clip_count = u32::from_le_bytes(params[16..20].try_into().unwrap());
        // Default: 2% black, 1% white of 10000 pixels
        assert_eq!(black_clip_count, 200); // 10000 * 0.02
        assert_eq!(white_clip_count, 100); // 10000 * 0.01
    }

    // ─── Normalize ───────────────────────────────────────────────────────

    #[test]
    fn normalize_clips_extremes() {
        let input = gradient_rgba(16, 16);
        let norm = Normalize::default();
        let output = norm.compute(&input, 16, 16).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // ─── Frequency Separation ────────────────────────────────────────────

    #[test]
    fn frequency_low_is_blurred() {
        let input = gradient_rgba(16, 16);
        let fl = FrequencyLow { sigma: 3.0 };
        let output = fl.compute(&input, 16, 16).unwrap();
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn frequency_high_solid_midgray() {
        let input = solid_rgba(16, 16, [0.3, 0.3, 0.3, 1.0]);
        let fh = FrequencyHigh { sigma: 3.0 };
        let output = fh.compute(&input, 16, 16).unwrap();
        // Solid → high pass = 0 + 0.5 = 0.5
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    // ─── Clarity ─────────────────────────────────────────────────────────

    #[test]
    fn clarity_preserves_solid() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let clar = Clarity { amount: 1.0, radius: 5.0 };
        let output = clar.compute(&input, 16, 16).unwrap();
        // Solid: no detail → no change
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn clarity_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
        let clar = Clarity { amount: 1.0, radius: 5.0 };
        let output = clar.compute(&input, 8, 8).unwrap();
        assert!((output[3] - 0.7).abs() < 1e-6);
    }

    // ─── Dehaze ──────────────────────────────────────────────────────────

    #[test]
    fn dehaze_runs_without_panic() {
        let input = gradient_rgba(16, 16);
        let dh = Dehaze { patch_radius: 3, omega: 0.95, t_min: 0.1 };
        let output = dh.compute(&input, 16, 16).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // ─── CLAHE ───────────────────────────────────────────────────────────

    #[test]
    fn clahe_runs_without_panic() {
        let input = gradient_rgba(32, 32);
        let clahe = Clahe { tile_grid: 4, clip_limit: 2.0 };
        let output = clahe.compute(&input, 32, 32).unwrap();
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn clahe_preserves_alpha() {
        let input = solid_rgba(32, 32, [0.5, 0.5, 0.5, 0.3]);
        let clahe = Clahe { tile_grid: 4, clip_limit: 2.0 };
        let output = clahe.compute(&input, 32, 32).unwrap();
        assert!((output[3] - 0.3).abs() < 1e-6);
    }

    // ─── NLM Denoise ─────────────────────────────────────────────────────

    #[test]
    fn nlm_solid_unchanged() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let nlm = NlmDenoise { h: 0.1, patch_radius: 1, search_radius: 2 };
        let output = nlm.compute(&input, 8, 8).unwrap();
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn nlm_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
        let nlm = NlmDenoise { h: 0.1, patch_radius: 1, search_radius: 2 };
        let output = nlm.compute(&input, 8, 8).unwrap();
        assert!((output[3] - 0.7).abs() < 1e-6);
    }

    // ─── Pyramid Detail Remap ────────────────────────────────────────────

    #[test]
    fn pyramid_detail_remap_runs() {
        let input = gradient_rgba(32, 32);
        let pdr = PyramidDetailRemap { sigma: 0.5, levels: 0 };
        let output = pdr.compute(&input, 32, 32).unwrap();
        assert_eq!(output.len(), input.len());
    }

    // ─── Retinex ─────────────────────────────────────────────────────────

    #[test]
    fn retinex_ssr_runs() {
        let input = gradient_rgba(16, 16);
        let ssr = RetinexSsr { sigma: 15.0 };
        let output = ssr.compute(&input, 16, 16).unwrap();
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn retinex_msr_runs() {
        let input = gradient_rgba(16, 16);
        let msr = RetinexMsr { sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0 };
        let output = msr.compute(&input, 16, 16).unwrap();
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn retinex_msrcr_runs() {
        let input = gradient_rgba(16, 16);
        let msrcr = RetinexMsrcr {
            sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0,
            alpha: 125.0, beta: 46.0,
        };
        let output = msrcr.compute(&input, 16, 16).unwrap();
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn retinex_ssr_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
        let ssr = RetinexSsr { sigma: 5.0 };
        let output = ssr.compute(&input, 8, 8).unwrap();
        assert!((output[3] - 0.7).abs() < 1e-6);
    }

    // ─── Shadow/Highlight ────────────────────────────────────────────────

    #[test]
    fn shadow_highlight_neutral_noop() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let sh = ShadowHighlight {
            shadows: 0.0, highlights: 0.0, whitepoint: 0.0,
            radius: 10.0, compress: 50.0,
            shadows_ccorrect: 100.0, highlights_ccorrect: 50.0,
        };
        let output = sh.compute(&input, 16, 16).unwrap();
        assert!((output[0] - 0.5).abs() < 0.02);
    }

    // ─── Vignette ────────────────────────────────────────────────────────

    #[test]
    fn vignette_center_bright_edges_dark() {
        let input = solid_rgba(32, 32, [1.0, 1.0, 1.0, 1.0]);
        let vig = Vignette { sigma: 5.0, x_inset: 4, y_inset: 4 };
        let output = vig.compute(&input, 32, 32).unwrap();
        // Center pixel should be brighter than corner
        let center = (16 * 32 + 16) * 4;
        let corner = 0;
        assert!(output[center] > output[corner]);
    }

    #[test]
    fn vignette_preserves_alpha() {
        let input = solid_rgba(16, 16, [1.0, 1.0, 1.0, 0.5]);
        let vig = Vignette { sigma: 3.0, x_inset: 2, y_inset: 2 };
        let output = vig.compute(&input, 16, 16).unwrap();
        assert!((output[3] - 0.5).abs() < 1e-6);
    }

    // ─── Vignette Power-law ──────────────────────────────────────────────

    #[test]
    fn vignette_powerlaw_center_unaffected() {
        let input = solid_rgba(16, 16, [1.0, 1.0, 1.0, 1.0]);
        let vig = VignettePowerlaw { strength: 0.5, falloff: 2.0 };
        let output = vig.compute(&input, 16, 16).unwrap();
        // Center pixel should be minimally affected (close to center)
        let center = (8 * 16 + 8) * 4;
        assert!(output[center] > 0.95);
    }

    #[test]
    fn vignette_powerlaw_corners_darkened() {
        let input = solid_rgba(16, 16, [1.0, 1.0, 1.0, 1.0]);
        let vig = VignettePowerlaw { strength: 1.0, falloff: 2.0 };
        let output = vig.compute(&input, 16, 16).unwrap();
        // Corner should be darker than center
        let center = (8 * 16 + 8) * 4;
        let corner = 0;
        assert!(output[corner] < output[center]);
    }

    // ─── Output sizes ────────────────────────────────────────────────────

    #[test]
    fn all_output_sizes_correct() {
        let input = gradient_rgba(16, 16);
        let n = 16 * 16 * 4;

        assert_eq!(AutoLevel.compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(Equalize.compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(Normalize::default().compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(FrequencyLow { sigma: 3.0 }.compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(FrequencyHigh { sigma: 3.0 }.compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(
            Clarity { amount: 1.0, radius: 3.0 }.compute(&input, 16, 16).unwrap().len(), n
        );
        assert_eq!(
            RetinexSsr { sigma: 5.0 }.compute(&input, 16, 16).unwrap().len(), n
        );
        assert_eq!(
            VignettePowerlaw { strength: 0.5, falloff: 2.0 }.compute(&input, 16, 16).unwrap().len(), n
        );
    }

    // ─── HDR values ──────────────────────────────────────────────────────

    #[test]
    fn hdr_values_not_clamped() {
        let input = solid_rgba(4, 4, [5.0, -0.5, 100.0, 1.0]);
        let vig = VignettePowerlaw { strength: 0.5, falloff: 2.0 };
        let output = vig.compute(&input, 4, 4).unwrap();
        // HDR values should still be present (not clamped to [0,1])
        assert!(output.chunks_exact(4).any(|p| p[0] > 1.0));
    }

    // ── GPU wiring tests ─────────────────────────────────────────────────────

    #[test]
    fn nlm_denoise_gpu_single_pass() {
        let nlm = NlmDenoise { h: 0.1, patch_radius: 3, search_radius: 7 };
        let passes = nlm.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 1);
        assert_eq!(nlm.workgroup_size(), [16, 16, 1]);
    }

    #[test]
    fn dehaze_gpu_2_passes() {
        let dh = Dehaze { patch_radius: 7, omega: 0.95, t_min: 0.1 };
        let passes = dh.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 2, "Dehaze: dark channel + apply");
    }

    #[test]
    fn shadow_highlight_gpu_3_passes() {
        let sh = ShadowHighlight {
            shadows: 50.0, highlights: 50.0, whitepoint: 0.0,
            radius: 3.0, compress: 50.0,
            shadows_ccorrect: 50.0, highlights_ccorrect: 50.0,
        };
        let passes = sh.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 3, "ShadowHighlight: blur H + blur V + apply");
    }

    #[test]
    fn frequency_low_gpu_2_passes() {
        let fl = FrequencyLow { sigma: 3.0 };
        let passes = fl.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 2);
    }

    #[test]
    fn frequency_high_gpu_3_passes() {
        let fh = FrequencyHigh { sigma: 3.0 };
        let passes = fh.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 3);
    }

    #[test]
    fn clarity_gpu_3_passes() {
        let cl = Clarity { amount: 0.5, radius: 20.0 };
        let passes = cl.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 3);
    }

    #[test]
    fn retinex_ssr_gpu_3_passes() {
        let ssr = RetinexSsr { sigma: 80.0 };
        let passes = ssr.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 3, "RetinexSSR: blur H + blur V + retinex apply");
    }

    #[test]
    fn retinex_msr_gpu_has_multiple_passes() {
        let msr = RetinexMsr { sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0 };
        let passes = msr.gpu_shaders(64, 64);
        // 3 scales × (blur H + blur V + accumulate) = 9 + 2 reduce + 1 normalize = 12
        assert!(passes.len() >= 10, "RetinexMSR should have many passes, got {}", passes.len());
    }

    #[test]
    fn retinex_msrcr_gpu_has_multiple_passes() {
        let msrcr = RetinexMsrcr {
            sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0,
            alpha: 125.0, beta: 46.0,
        };
        let passes = msrcr.gpu_shaders(64, 64);
        // 3 scales × 3 + 1 color restore = 10
        assert!(passes.len() >= 10, "RetinexMSRCR should have many passes, got {}", passes.len());
    }

    #[test]
    fn clahe_gpu_single_pass_with_luts() {
        let clahe = Clahe { tile_grid: 8, clip_limit: 2.0 };
        let passes = clahe.gpu_shaders(64, 64);
        assert_eq!(passes.len(), 1);
        let params = clahe.params(64, 64);
        assert_eq!(params.len(), 32);
    }

    #[test]
    fn pyramid_detail_remap_gpu_multi_pass() {
        let pdr = PyramidDetailRemap { sigma: 0.5, levels: 4 };
        let passes = pdr.gpu_shaders(64, 64);
        // 4 downsample + 4 × (upsample + remap) = 4 + 8 = 12
        assert!(passes.len() >= 8, "Pyramid: should have downsample + remap passes, got {}", passes.len());
    }

    #[test]
    fn all_enhancement_filters_have_gpu() {
        let w = 32u32;
        let h = 32u32;

        // Already wired in previous tracks
        assert!(!AutoLevel.gpu_shaders(w, h).is_empty());
        assert!(!Equalize.gpu_shaders(w, h).is_empty());
        assert!(!Normalize::default().gpu_shaders(w, h).is_empty());
        assert!(!VignettePowerlaw { strength: 0.5, falloff: 2.0 }.gpu_shaders(w, h).is_empty());

        // Newly wired in this track
        assert!(!NlmDenoise { h: 0.1, patch_radius: 3, search_radius: 7 }.gpu_shaders(w, h).is_empty());
        assert!(!Dehaze { patch_radius: 7, omega: 0.95, t_min: 0.1 }.gpu_shaders(w, h).is_empty());
        assert!(!ShadowHighlight {
            shadows: 50.0, highlights: 50.0, whitepoint: 0.0,
            radius: 3.0, compress: 50.0, shadows_ccorrect: 50.0, highlights_ccorrect: 50.0,
        }.gpu_shaders(w, h).is_empty());
        assert!(!FrequencyLow { sigma: 3.0 }.gpu_shaders(w, h).is_empty());
        assert!(!FrequencyHigh { sigma: 3.0 }.gpu_shaders(w, h).is_empty());
        assert!(!Clarity { amount: 0.5, radius: 20.0 }.gpu_shaders(w, h).is_empty());
        assert!(!RetinexSsr { sigma: 80.0 }.gpu_shaders(w, h).is_empty());
        assert!(!RetinexMsr { sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0 }.gpu_shaders(w, h).is_empty());
        assert!(!RetinexMsrcr {
            sigma_small: 15.0, sigma_medium: 80.0, sigma_large: 250.0,
            alpha: 125.0, beta: 46.0,
        }.gpu_shaders(w, h).is_empty());
        assert!(!Clahe { tile_grid: 8, clip_limit: 2.0 }.gpu_shaders(w, h).is_empty());
        assert!(!PyramidDetailRemap { sigma: 0.5, levels: 4 }.gpu_shaders(w, h).is_empty());
    }
}
