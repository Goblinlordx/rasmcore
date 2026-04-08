//! Effect filters — creative/stylistic operations on f32 pixel data.
//!
//! All operate on `&[f32]` RGBA (4 channels per pixel). No format dispatch.
//! No u8/u16 paths. Just f32.
//!
//! Includes: noise (Gaussian, uniform, salt-pepper, Poisson), film grain,
//! pixelate, halftone, oil paint, emboss, charcoal, glitch, chromatic
//! aberration/split, light leak, mirror kaleidoscope, vignette effects.
//!
//! Solarize is in adjustment.rs (point op with AnalyticOp support).

use crate::filters::spatial::GaussianBlur;
use crate::node::PipelineError;
use crate::noise;
use crate::ops::{Filter, GpuFilter};

use super::helpers::{gpu_params_wh, luminance};

// PRNG and noise use the shared noise module (crate::noise).
use noise::{Rng, SEED_GAUSSIAN_NOISE, SEED_UNIFORM_NOISE, SEED_SALT_PEPPER, SEED_POISSON_NOISE, SEED_GLITCH};

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
// Noise filters
// ═══════════════════════════════════════════════════════════════════════════════

/// Gaussian noise — additive normally-distributed noise.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "gaussian_noise", category = "effect", cost = "O(n)")]
pub struct GaussianNoise {
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub amount: f32,
    #[param(min = -255.0, max = 255.0, default = 0.0)]
    pub mean: f32,
    #[param(min = 0.0, max = 255.0, default = 25.0)]
    pub sigma: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for GaussianNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.amount <= 0.0 {
            return Ok(input.to_vec());
        }
        let amount = self.amount / 100.0;
        let sigma = self.sigma / 255.0; // normalize to [0,1] range
        let mean = self.mean / 255.0;
        let mut rng = Rng::with_offset(self.seed, SEED_GAUSSIAN_NOISE);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            let n0 = (mean + sigma * rng.next_gaussian()) * amount;
            let n1 = (mean + sigma * rng.next_gaussian()) * amount;
            let n2 = (mean + sigma * rng.next_gaussian()) * amount;
            pixel[0] += n0;
            pixel[1] += n1;
            pixel[2] += n2;
        }
        Ok(out)
    }
}

/// Uniform noise — additive uniformly-distributed noise.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "uniform_noise", category = "effect", cost = "O(n)")]
pub struct UniformNoise {
    #[param(min = 0.0, max = 255.0, default = 25.0)]
    pub range: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for UniformNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.range <= 0.0 {
            return Ok(input.to_vec());
        }
        let range = self.range / 255.0;
        let mut rng = Rng::with_offset(self.seed, SEED_UNIFORM_NOISE);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            pixel[0] += rng.next_f32_signed() * range;
            pixel[1] += rng.next_f32_signed() * range;
            pixel[2] += rng.next_f32_signed() * range;
        }
        Ok(out)
    }
}

/// Salt-and-pepper noise — randomly replace pixels with black or white.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "salt_pepper_noise", category = "effect", cost = "O(n)")]
pub struct SaltPepperNoise {
    #[param(min = 0.0, max = 1.0, default = 0.05)]
    pub density: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for SaltPepperNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut rng = Rng::with_offset(self.seed, SEED_SALT_PEPPER);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            if rng.next_f32() < self.density {
                let val = if rng.next_f32() < 0.5 { 0.0 } else { 1.0 };
                pixel[0] = val;
                pixel[1] = val;
                pixel[2] = val;
            }
        }
        Ok(out)
    }
}

/// Poisson noise — signal-dependent noise (brighter regions get more).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "poisson_noise", category = "effect", cost = "O(n)")]
pub struct PoissonNoise {
    #[param(min = 0.1, max = 1000.0, default = 100.0)]
    pub scale: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for PoissonNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.scale <= 0.0 {
            return Ok(input.to_vec());
        }
        let mut rng = Rng::with_offset(self.seed, SEED_POISSON_NOISE);
        let mut out = input.to_vec();
        let inv_scale = 1.0 / self.scale;

        for pixel in out.chunks_exact_mut(4) {
            #[allow(clippy::needless_range_loop)]
            for c in 0..3 {
                let lambda = (pixel[c].max(0.0) * self.scale).max(0.001);
                // Poisson approximation via Gaussian for lambda > 10
                let noisy = if lambda > 10.0 {
                    lambda + lambda.sqrt() * rng.next_gaussian()
                } else {
                    // Knuth algorithm for small lambda
                    let l = (-lambda).exp();
                    let mut k = 0.0f32;
                    let mut p = 1.0f32;
                    loop {
                        k += 1.0;
                        p *= rng.next_f32();
                        if p <= l {
                            break;
                        }
                    }
                    k - 1.0
                };
                pixel[c] = noisy * inv_scale;
            }
        }
        Ok(out)
    }
}

/// Film grain — photographic grain overlay with noise texture.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "film_grain", category = "effect", cost = "O(n)")]
pub struct FilmGrain {
    #[param(min = 0.0, max = 1.0, default = 0.1)]
    pub amount: f32,
    #[param(min = 1.0, max = 10.0, default = 1.0)]
    pub size: f32,
    #[param(default = 42)]
    pub seed: u64,
}

impl Filter for FilmGrain {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.amount <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let mut rng = Rng::with_offset(self.seed, noise::SEED_FILM_GRAIN);

        // Generate grain at reduced resolution then upsample
        let grain_w = (w as f32 / self.size.max(1.0)).ceil() as usize;
        let grain_h = (h as f32 / self.size.max(1.0)).ceil() as usize;
        let mut grain = vec![0.0f32; grain_w.max(1) * grain_h.max(1)];
        for v in &mut grain {
            *v = rng.next_gaussian() * self.amount;
        }

        let mut out = input.to_vec();
        let inv_size = 1.0 / self.size.max(1.0);

        for y in 0..h {
            for x in 0..w {
                let gx = ((x as f32 * inv_size) as usize).min(grain_w.saturating_sub(1));
                let gy = ((y as f32 * inv_size) as usize).min(grain_h.saturating_sub(1));
                let g = grain[gy * grain_w.max(1) + gx];

                let idx = (y * w + x) * 4;
                // Grain weighted by luminance (more visible in midtones)
                let luma = luminance(out[idx], out[idx + 1], out[idx + 2]).clamp(0.0, 1.0);
                let weight = 4.0 * luma * (1.0 - luma); // midtone peak
                for c in 0..3 {
                    out[idx + c] += g * weight;
                }
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pixel / block effects
// ═══════════════════════════════════════════════════════════════════════════════

/// Pixelate — block-grid mosaic with mean color per cell.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "pixelate", category = "effect", cost = "O(n)")]
pub struct Pixelate {
    #[param(min = 1, max = 100, default = 8)]
    pub block_size: u32,
}

impl Filter for Pixelate {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let bs = self.block_size.max(1) as usize;
        let mut out = input.to_vec();

        let mut by = 0;
        while by < h {
            let bh = bs.min(h - by);
            let mut bx = 0;
            while bx < w {
                let bw = bs.min(w - bx);
                let count = (bw * bh) as f32;

                // Compute mean color
                let mut sum = [0.0f32; 3];
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + (bx + dx)) * 4;
                        sum[0] += input[idx];
                        sum[1] += input[idx + 1];
                        sum[2] += input[idx + 2];
                    }
                }
                let mean = [sum[0] / count, sum[1] / count, sum[2] / count];

                // Fill block
                for dy in 0..bh {
                    for dx in 0..bw {
                        let idx = ((by + dy) * w + (bx + dx)) * 4;
                        out[idx] = mean[0];
                        out[idx + 1] = mean[1];
                        out[idx + 2] = mean[2];
                    }
                }

                bx += bs;
            }
            by += bs;
        }
        Ok(out)
    }
}

/// Halftone — CMYK-style dot pattern via sine-wave screening.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "halftone", category = "effect", cost = "O(n)")]
pub struct Halftone {
    #[param(min = 1.0, max = 50.0, default = 8.0)]
    pub dot_size: f32,
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle_offset: f32,
}

impl Filter for Halftone {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let ds = self.dot_size.max(1.0);
        let freq = std::f32::consts::PI / ds;

        // CMYK screen angles
        let angles = [
            (15.0 + self.angle_offset).to_radians(),
            (75.0 + self.angle_offset).to_radians(),
            (0.0 + self.angle_offset).to_radians(),
            (45.0 + self.angle_offset).to_radians(),
        ];

        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let r = input[idx];
                let g = input[idx + 1];
                let b = input[idx + 2];

                // Simple RGB to CMYK
                let k = 1.0 - r.max(g).max(b);
                let inv_k = if k < 1.0 { 1.0 / (1.0 - k) } else { 0.0 };
                let c = (1.0 - r - k) * inv_k;
                let m = (1.0 - g - k) * inv_k;
                let yc = (1.0 - b - k) * inv_k;

                let cmyk = [c, m, yc, k];
                let mut screened = [0.0f32; 4];

                for (i, &angle) in angles.iter().enumerate() {
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let rx = x as f32 * cos_a + y as f32 * sin_a;
                    let ry = -(x as f32) * sin_a + y as f32 * cos_a;
                    let screen = (rx * freq).sin() * (ry * freq).sin();
                    let threshold = (screen + 1.0) * 0.5;
                    screened[i] = if cmyk[i] > threshold { 1.0 } else { 0.0 };
                }

                // CMYK back to RGB
                let inv_k2 = 1.0 - screened[3];
                out[idx] = (1.0 - screened[0]) * inv_k2;
                out[idx + 1] = (1.0 - screened[1]) * inv_k2;
                out[idx + 2] = (1.0 - screened[2]) * inv_k2;
            }
        }
        Ok(out)
    }
}

/// Oil paint — neighborhood mode filter (most frequent intensity bin).
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "oil_paint", category = "effect", cost = "O(n * radius^2)")]
pub struct OilPaint {
    #[param(min = 1, max = 20, default = 4)]
    pub radius: u32,
}

impl Filter for OilPaint {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as usize;
        const BINS: usize = 256;
        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let mut count = [0u32; BINS];
                let mut sum_r = [0.0f32; BINS];
                let mut sum_g = [0.0f32; BINS];
                let mut sum_b = [0.0f32; BINS];

                let y0 = y.saturating_sub(r);
                let y1 = (y + r + 1).min(h);
                let x0 = x.saturating_sub(r);
                let x1 = (x + r + 1).min(w);

                for ny in y0..y1 {
                    for nx in x0..x1 {
                        let idx = (ny * w + nx) * 4;
                        let intensity = (luminance(input[idx], input[idx + 1], input[idx + 2])
                            .clamp(0.0, 1.0)
                            * 255.0) as usize;
                        let bin = intensity.min(BINS - 1);
                        count[bin] += 1;
                        sum_r[bin] += input[idx];
                        sum_g[bin] += input[idx + 1];
                        sum_b[bin] += input[idx + 2];
                    }
                }

                // Find most frequent bin
                let max_bin = count
                    .iter()
                    .enumerate()
                    .max_by_key(|&(_, c)| *c)
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let cnt = count[max_bin].max(1) as f32;
                let idx = (y * w + x) * 4;
                out[idx] = sum_r[max_bin] / cnt;
                out[idx + 1] = sum_g[max_bin] / cnt;
                out[idx + 2] = sum_b[max_bin] / cnt;
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Emboss / Charcoal
// ═══════════════════════════════════════════════════════════════════════════════

/// Emboss — 3D relief effect via directional convolution kernel.
///
/// `output = convolve(input, emboss_kernel) + 0.5`
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "emboss", category = "effect", cost = "O(n)")]
pub struct Emboss;

impl Filter for Emboss {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        // Standard emboss kernel (top-left to bottom-right)
        let kernel: [f32; 9] = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 3];
                for ky in 0..3i32 {
                    for kx in 0..3i32 {
                        let sx = clamp_coord(x as i32 + kx - 1, w);
                        let sy = clamp_coord(y as i32 + ky - 1, h);
                        let k = kernel[(ky * 3 + kx) as usize];
                        let idx = (sy * w + sx) * 4;
                        for c in 0..3 {
                            sum[c] += k * input[idx + c];
                        }
                    }
                }
                let idx = (y * w + x) * 4;
                for c in 0..3 {
                    out[idx + c] = sum[c] + 0.5; // neutral gray offset
                }
                out[idx + 3] = input[idx + 3]; // alpha
            }
        }
        Ok(out)
    }
}

/// Charcoal — edge detection → blur → invert for pencil sketch effect.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "charcoal", category = "effect", cost = "O(n * radius) via gaussian_blur")]
pub struct Charcoal {
    #[param(min = 0.0, max = 20.0, default = 1.0)]
    pub radius: f32,
    #[param(min = 0.0, max = 20.0, default = 1.0)]
    pub sigma: f32,
}

impl Filter for Charcoal {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;

        // Step 1: Sobel edge detection (on luminance)
        let sobel_x: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut edges = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut gx = 0.0f32;
                let mut gy = 0.0f32;
                for ky in 0..3i32 {
                    for kx in 0..3i32 {
                        let sx = clamp_coord(x as i32 + kx - 1, w);
                        let sy = clamp_coord(y as i32 + ky - 1, h);
                        let idx = (sy * w + sx) * 4;
                        let luma = luminance(input[idx], input[idx + 1], input[idx + 2]);
                        let ki = (ky * 3 + kx) as usize;
                        gx += sobel_x[ki] * luma;
                        gy += sobel_y[ki] * luma;
                    }
                }
                let mag = (gx * gx + gy * gy).sqrt().min(1.0);
                let idx = (y * w + x) * 4;
                edges[idx] = mag;
                edges[idx + 1] = mag;
                edges[idx + 2] = mag;
                edges[idx + 3] = input[idx + 3];
            }
        }

        // Step 2: Blur edges
        let blur = GaussianBlur { radius: self.sigma };
        let blurred = blur.compute(&edges, width, height)?;

        // Step 3: Invert for charcoal look (dark lines on white)
        let mut out = Vec::with_capacity(blurred.len());
        for (i, &v) in blurred.iter().enumerate() {
            if i % 4 == 3 {
                out.push(v); // alpha
            } else {
                out.push(1.0 - v);
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Chromatic effects
// ═══════════════════════════════════════════════════════════════════════════════

/// Chromatic split — offset RGB channels independently.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "chromatic_split", category = "effect", cost = "O(n)")]
pub struct ChromaticSplit {
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub red_dx: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub red_dy: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub green_dx: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub green_dy: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub blue_dx: f32,
    #[param(min = -100.0, max = 100.0, default = 0.0)]
    pub blue_dy: f32,
}

impl Filter for ChromaticSplit {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;

                // Red channel from offset position
                let rx = clamp_coord((x as f32 + self.red_dx).round() as i32, w);
                let ry = clamp_coord((y as f32 + self.red_dy).round() as i32, h);
                out[idx] = input[(ry * w + rx) * 4];

                // Green channel
                let gx = clamp_coord((x as f32 + self.green_dx).round() as i32, w);
                let gy = clamp_coord((y as f32 + self.green_dy).round() as i32, h);
                out[idx + 1] = input[(gy * w + gx) * 4 + 1];

                // Blue channel
                let bx = clamp_coord((x as f32 + self.blue_dx).round() as i32, w);
                let by = clamp_coord((y as f32 + self.blue_dy).round() as i32, h);
                out[idx + 2] = input[(by * w + bx) * 4 + 2];

                out[idx + 3] = input[idx + 3]; // alpha
            }
        }
        Ok(out)
    }
}

/// Chromatic aberration — radial R/B channel displacement.
///
/// R channel shifts away from center, B channel shifts toward center.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "chromatic_aberration", category = "effect", cost = "O(n)")]
pub struct ChromaticAberration {
    #[param(min = 0.0, max = 50.0, default = 5.0)]
    pub strength: f32,
}

impl Filter for ChromaticAberration {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt().max(1.0);
        let strength = self.strength / max_dist;

        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 4;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();
                let shift = dist * strength;

                // Red: shift outward
                let rx = clamp_coord((x as f32 + dx * shift / dist.max(1.0)).round() as i32, w);
                let ry = clamp_coord((y as f32 + dy * shift / dist.max(1.0)).round() as i32, h);
                out[idx] = input[(ry * w + rx) * 4];

                // Green: no shift
                out[idx + 1] = input[idx + 1];

                // Blue: shift inward
                let bx = clamp_coord((x as f32 - dx * shift / dist.max(1.0)).round() as i32, w);
                let by = clamp_coord((y as f32 - dy * shift / dist.max(1.0)).round() as i32, h);
                out[idx + 2] = input[(by * w + bx) * 4 + 2];

                out[idx + 3] = input[idx + 3];
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Glitch / Light Leak / Mirror Kaleidoscope
// ═══════════════════════════════════════════════════════════════════════════════

/// Glitch — horizontal scanline displacement with RGB channel offset.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "glitch", category = "effect", cost = "O(n)")]
pub struct Glitch {
    #[param(min = 0.0, max = 200.0, default = 20.0)]
    pub shift_amount: f32,
    #[param(min = 0.0, max = 100.0, default = 10.0)]
    pub channel_offset: f32,
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub intensity: f32,
    #[param(min = 1, max = 100, default = 8)]
    pub band_height: u32,
    #[param(min = 0, max = 100, default = 42)]
    pub seed: u32,
}

impl Filter for Glitch {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let band_h = self.band_height.max(1) as usize;
        let mut out = input.to_vec();

        for y in 0..h {
            let band = y / band_h;
            let noise = noise::noise_1d(band as u32, self.seed as u64 ^ SEED_GLITCH);

            if noise.abs() > 1.0 - self.intensity {
                let shift = (noise * self.shift_amount) as i32;
                let ch_off = (noise * self.channel_offset) as i32;

                for x in 0..w {
                    let idx = (y * w + x) * 4;

                    // Red channel with shift + offset
                    let rx = clamp_coord(x as i32 + shift + ch_off, w);
                    out[idx] = input[(y * w + rx) * 4];

                    // Green channel with shift
                    let gx = clamp_coord(x as i32 + shift, w);
                    out[idx + 1] = input[(y * w + gx) * 4 + 1];

                    // Blue channel with shift - offset
                    let bx = clamp_coord(x as i32 + shift - ch_off, w);
                    out[idx + 2] = input[(y * w + bx) * 4 + 2];
                }
            }
        }
        Ok(out)
    }
}

/// Light leak — procedural warm-toned radial gradient with screen blend.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "light_leak", category = "effect", cost = "O(n)")]
pub struct LightLeak {
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub intensity: f32,
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub position_x: f32,
    #[param(min = 0.0, max = 1.0, default = 0.5)]
    pub position_y: f32,
    #[param(min = 0.0, max = 2.0, default = 0.5)]
    pub radius: f32,
    #[param(min = 0.0, max = 1.0, default = 0.8)]
    pub warmth: f32,
}

impl Filter for LightLeak {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let cx = self.position_x * w as f32;
        let cy = self.position_y * h as f32;
        let max_r = self.radius * (w.max(h) as f32);
        let inv_r = if max_r > 0.0 { 1.0 / max_r } else { 0.0 };

        let mut out = input.to_vec();

        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt() * inv_r;
                let falloff = (1.0 - dist).max(0.0);
                let falloff = falloff * falloff; // quadratic

                // Warm-toned leak color (orangeish)
                let leak_r = 1.0 * self.warmth + 0.8 * (1.0 - self.warmth);
                let leak_g = 0.6 * self.warmth + 0.4 * (1.0 - self.warmth);
                let leak_b = 0.2 * self.warmth + 0.1 * (1.0 - self.warmth);

                let strength = falloff * self.intensity;
                let idx = (y * w + x) * 4;

                // Screen blend: a + b - a*b
                out[idx] = out[idx] + leak_r * strength - out[idx] * leak_r * strength;
                out[idx + 1] = out[idx + 1] + leak_g * strength - out[idx + 1] * leak_g * strength;
                out[idx + 2] = out[idx + 2] + leak_b * strength - out[idx + 2] * leak_b * strength;
            }
        }
        Ok(out)
    }
}

/// Mirror kaleidoscope — reflect/mirror segments around axis.
#[derive(Clone, rasmcore_macros::V2Filter)]
#[filter(name = "mirror_kaleidoscope", category = "effect", cost = "O(n)")]
pub struct MirrorKaleidoscope {
    #[param(min = 2, max = 32, default = 4)]
    pub segments: u32,
    #[param(min = 0.0, max = 360.0, default = 0.0)]
    pub angle: f32,
    #[param(min = 0, max = 2, default = 0)]
    pub mode: u32, // 0=horizontal, 1=vertical, 2=angular
}

impl Filter for MirrorKaleidoscope {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let segments = self.segments.max(2) as usize;
        let mut out = vec![0.0f32; w * h * 4];

        match self.mode {
            0 => {
                // Horizontal mirror
                let seg_w = w / segments;
                if seg_w == 0 {
                    return Ok(input.to_vec());
                }
                for y in 0..h {
                    for x in 0..w {
                        let seg = x / seg_w;
                        let local_x = x % seg_w;
                        let src_x = if seg.is_multiple_of(2) { local_x } else { seg_w - 1 - local_x };
                        let src_x = src_x.min(seg_w - 1);
                        let src_idx = (y * w + src_x) * 4;
                        let dst_idx = (y * w + x) * 4;
                        out[dst_idx..dst_idx + 4].copy_from_slice(&input[src_idx..src_idx + 4]);
                    }
                }
            }
            1 => {
                // Vertical mirror
                let seg_h = h / segments;
                if seg_h == 0 {
                    return Ok(input.to_vec());
                }
                for y in 0..h {
                    let seg = y / seg_h;
                    let local_y = y % seg_h;
                    let src_y = if seg.is_multiple_of(2) { local_y } else { seg_h - 1 - local_y };
                    let src_y = src_y.min(seg_h - 1);
                    for x in 0..w {
                        let src_idx = (src_y * w + x) * 4;
                        let dst_idx = (y * w + x) * 4;
                        out[dst_idx..dst_idx + 4].copy_from_slice(&input[src_idx..src_idx + 4]);
                    }
                }
            }
            _ => {
                // Angular kaleidoscope
                let cx = w as f32 / 2.0;
                let cy = h as f32 / 2.0;
                let segment_angle = std::f32::consts::TAU / segments as f32;
                let base_angle = self.angle.to_radians();

                for y in 0..h {
                    for x in 0..w {
                        let dx = x as f32 - cx;
                        let dy = y as f32 - cy;
                        let mut angle = dy.atan2(dx) - base_angle;
                        if angle < 0.0 {
                            angle += std::f32::consts::TAU;
                        }

                        let seg = (angle / segment_angle) as usize;
                        let local_angle = angle - seg as f32 * segment_angle;
                        let mapped = if seg.is_multiple_of(2) {
                            local_angle + base_angle
                        } else {
                            (segment_angle - local_angle) + base_angle
                        };

                        let dist = (dx * dx + dy * dy).sqrt();
                        let src_x = (cx + dist * mapped.cos()).round() as i32;
                        let src_y = (cy + dist * mapped.sin()).round() as i32;

                        let src_x = clamp_coord(src_x, w);
                        let src_y = clamp_coord(src_y, h);

                        let src_idx = (src_y * w + src_x) * 4;
                        let dst_idx = (y * w + x) * 4;
                        out[dst_idx..dst_idx + 4].copy_from_slice(&input[src_idx..src_idx + 4]);
                    }
                }
            }
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU Shaders — GpuFilter implementations for per-pixel effect filters
// ═══════════════════════════════════════════════════════════════════════════════

fn gpu_params_push_f32(buf: &mut Vec<u8>, v: f32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn gpu_params_push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

// ── Gaussian Noise GPU ─────────────────────────────────────────────────────

const GAUSSIAN_NOISE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  amount: f32,
  mean: f32,
  sigma: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  // Box-Muller: two uniform -> one gaussian
  let u1 = max(abs(noise_2d(idx, 0u, params.seed_lo, params.seed_hi) * 0.5 + 0.5), 0.00001);
  let u2 = noise_2d(idx, 1u, params.seed_lo + 7u, params.seed_hi) * 0.5 + 0.5;
  let g = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
  let n = (params.mean + params.sigma * g) * params.amount;
  store_pixel(idx, vec4<f32>(pixel.x + n, pixel.y + n, pixel.z + n, pixel.w));
}
"#;

impl GpuFilter for GaussianNoise {
    fn shader_body(&self) -> &str { GAUSSIAN_NOISE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_GAUSSIAN_NOISE;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.amount / 100.0);
        gpu_params_push_f32(&mut buf, self.mean / 255.0);
        gpu_params_push_f32(&mut buf, self.sigma / 255.0);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0); // pad
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, GAUSSIAN_NOISE_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}

// ── Uniform Noise GPU ──────────────────────────────────────────────────────

const UNIFORM_NOISE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  range: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  let n = noise_2d(idx, 0u, params.seed_lo, params.seed_hi) * params.range;
  store_pixel(idx, vec4<f32>(pixel.x + n, pixel.y + n, pixel.z + n, pixel.w));
}
"#;

impl GpuFilter for UniformNoise {
    fn shader_body(&self) -> &str { UNIFORM_NOISE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_UNIFORM_NOISE;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.range / 255.0);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, UNIFORM_NOISE_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}

// ── Salt & Pepper Noise GPU ────────────────────────────────────────────────

const SALT_PEPPER_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  density: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  let r = noise_2d(idx, 0u, params.seed_lo, params.seed_hi) * 0.5 + 0.5;
  if (r < params.density) {
    let bw = noise_2d(idx, 1u, params.seed_lo + 3u, params.seed_hi) * 0.5 + 0.5;
    let val = select(0.0, 1.0, bw > 0.5);
    store_pixel(idx, vec4<f32>(val, val, val, pixel.w));
  } else {
    store_pixel(idx, pixel);
  }
}
"#;

impl GpuFilter for SaltPepperNoise {
    fn shader_body(&self) -> &str { SALT_PEPPER_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_SALT_PEPPER;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.density);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, SALT_PEPPER_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}

// ── Poisson Noise GPU ──────────────────────────────────────────────────────

const POISSON_NOISE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  scale: f32,
  inv_scale: f32,
  seed_lo: u32,
  seed_hi: u32,
  _pad0: u32,
  _pad1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.width * params.height) { return; }
  let pixel = load_pixel(idx);
  // Gaussian approx for Poisson: noisy = lambda + sqrt(lambda) * gaussian
  var result = pixel;
  for (var c = 0u; c < 3u; c = c + 1u) {
    let val = select(pixel.x, select(pixel.y, pixel.z, c == 2u), c >= 1u);
    let lambda = max(val, 0.0) * params.scale;
    let u1 = max(abs(noise_2d(idx * 3u + c, 0u, params.seed_lo, params.seed_hi) * 0.5 + 0.5), 0.00001);
    let u2 = noise_2d(idx * 3u + c, 1u, params.seed_lo + 5u, params.seed_hi) * 0.5 + 0.5;
    let g = sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
    let noisy = max(lambda + sqrt(max(lambda, 0.001)) * g, 0.0) * params.inv_scale;
    switch c {
      case 0u: { result.x = noisy; }
      case 1u: { result.y = noisy; }
      case 2u: { result.z = noisy; }
      default: {}
    }
  }
  store_pixel(idx, result);
}
"#;

impl GpuFilter for PoissonNoise {
    fn shader_body(&self) -> &str { POISSON_NOISE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [256, 1, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed ^ noise::SEED_POISSON_NOISE;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.scale);
        gpu_params_push_f32(&mut buf, 1.0 / self.scale.max(0.001));
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, POISSON_NOISE_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}

// ── Light Leak GPU ─────────────────────────────────────────────────────────

const LIGHT_LEAK_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  intensity: f32,
  pos_x: f32,
  pos_y: f32,
  radius: f32,
  warmth: f32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let nx = f32(gid.x) / f32(params.width);
  let ny = f32(gid.y) / f32(params.height);
  let dx = nx - params.pos_x;
  let dy = ny - params.pos_y;
  let dist = sqrt(dx * dx + dy * dy);
  let falloff = max(1.0 - dist / max(params.radius, 0.001), 0.0);
  let leak = falloff * falloff * params.intensity;
  // Screen blend: 1 - (1-a)*(1-b)
  let lr = 1.0 * params.warmth;
  let lg = 0.8 * params.warmth;
  let lb = 0.3 * params.warmth;
  let r = 1.0 - (1.0 - pixel.x) * (1.0 - lr * leak);
  let g = 1.0 - (1.0 - pixel.y) * (1.0 - lg * leak);
  let b = 1.0 - (1.0 - pixel.z) * (1.0 - lb * leak);
  store_pixel(idx, vec4<f32>(r, g, b, pixel.w));
}
"#;

impl GpuFilter for LightLeak {
    fn shader_body(&self) -> &str { LIGHT_LEAK_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.intensity);
        gpu_params_push_f32(&mut buf, self.position_x);
        gpu_params_push_f32(&mut buf, self.position_y);
        gpu_params_push_f32(&mut buf, self.radius);
        gpu_params_push_f32(&mut buf, self.warmth);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}

// ── Glitch GPU ─────────────────────────────────────────────────────────────

const GLITCH_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  shift_amount: f32,
  channel_offset: f32,
  intensity: f32,
  band_height: u32,
  seed_lo: u32,
  seed_hi: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let pixel = load_pixel(idx);
  let band = gid.y / max(params.band_height, 1u);
  let n = noise_2d(band, 0u, params.seed_lo, params.seed_hi);
  if (abs(n) <= 1.0 - params.intensity) {
    store_pixel(idx, pixel);
    return;
  }
  let shift = i32(n * params.shift_amount);
  let ch_off = i32(n * params.channel_offset);
  let w = i32(params.width);
  let rx = clamp(i32(gid.x) + shift + ch_off, 0, w - 1);
  let gx = clamp(i32(gid.x) + shift, 0, w - 1);
  let bx = clamp(i32(gid.x) + shift - ch_off, 0, w - 1);
  let rp = load_pixel(gid.y * params.width + u32(rx));
  let gp = load_pixel(gid.y * params.width + u32(gx));
  let bp = load_pixel(gid.y * params.width + u32(bx));
  store_pixel(idx, vec4<f32>(rp.x, gp.y, bp.z, pixel.w));
}
"#;

impl GpuFilter for Glitch {
    fn shader_body(&self) -> &str { GLITCH_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let seed = self.seed as u64 ^ noise::SEED_GLITCH;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.shift_amount);
        gpu_params_push_f32(&mut buf, self.channel_offset);
        gpu_params_push_f32(&mut buf, self.intensity);
        gpu_params_push_u32(&mut buf, self.band_height);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, GLITCH_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}

// ── Chromatic Split GPU ────────────────────────────────────────────────────

const CHROMATIC_SPLIT_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  red_dx: f32,
  red_dy: f32,
  green_dx: f32,
  green_dy: f32,
  blue_dx: f32,
  blue_dy: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let w = i32(params.width);
  let h = i32(params.height);
  let rx = clamp(i32(gid.x) + i32(params.red_dx), 0, w - 1);
  let ry = clamp(i32(gid.y) + i32(params.red_dy), 0, h - 1);
  let gx = clamp(i32(gid.x) + i32(params.green_dx), 0, w - 1);
  let gy = clamp(i32(gid.y) + i32(params.green_dy), 0, h - 1);
  let bx = clamp(i32(gid.x) + i32(params.blue_dx), 0, w - 1);
  let by = clamp(i32(gid.y) + i32(params.blue_dy), 0, h - 1);
  let rp = load_pixel(u32(ry) * params.width + u32(rx));
  let gp = load_pixel(u32(gy) * params.width + u32(gx));
  let bp = load_pixel(u32(by) * params.width + u32(bx));
  let pixel = load_pixel(idx);
  store_pixel(idx, vec4<f32>(rp.x, gp.y, bp.z, pixel.w));
}
"#;

impl GpuFilter for ChromaticSplit {
    fn shader_body(&self) -> &str { CHROMATIC_SPLIT_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.red_dx);
        gpu_params_push_f32(&mut buf, self.red_dy);
        gpu_params_push_f32(&mut buf, self.green_dx);
        gpu_params_push_f32(&mut buf, self.green_dy);
        gpu_params_push_f32(&mut buf, self.blue_dx);
        gpu_params_push_f32(&mut buf, self.blue_dy);
        buf
    }
}

// ── Chromatic Aberration GPU ───────────────────────────────────────────────

const CHROMATIC_ABERRATION_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  strength: f32,
  _pad: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  let cx = f32(params.width) / 2.0;
  let cy = f32(params.height) / 2.0;
  let dx = f32(gid.x) - cx;
  let dy = f32(gid.y) - cy;
  let max_d = sqrt(cx * cx + cy * cy);
  let dist = sqrt(dx * dx + dy * dy);
  let shift = params.strength * dist / max(max_d, 1.0);
  let dir_x = select(dx / dist, 0.0, dist < 0.001);
  let dir_y = select(dy / dist, 0.0, dist < 0.001);
  let w = i32(params.width);
  let h = i32(params.height);
  // Red: shift outward
  let rrx = clamp(i32(f32(gid.x) + dir_x * shift), 0, w - 1);
  let rry = clamp(i32(f32(gid.y) + dir_y * shift), 0, h - 1);
  // Blue: shift inward
  let brx = clamp(i32(f32(gid.x) - dir_x * shift), 0, w - 1);
  let bry = clamp(i32(f32(gid.y) - dir_y * shift), 0, h - 1);
  let rp = load_pixel(u32(rry) * params.width + u32(rrx));
  let pixel = load_pixel(idx);
  let bp = load_pixel(u32(bry) * params.width + u32(brx));
  store_pixel(idx, vec4<f32>(rp.x, pixel.y, bp.z, pixel.w));
}
"#;

impl GpuFilter for ChromaticAberration {
    fn shader_body(&self) -> &str { CHROMATIC_ABERRATION_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.strength);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}

// ── Mirror Kaleidoscope GPU ────────────────────────────────────────────────

const MIRROR_KALEIDOSCOPE_WGSL: &str = r#"
struct Params {
  width: u32,
  height: u32,
  segments: u32,
  angle: f32,
  mode: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let idx = gid.y * params.width + gid.x;
  var sx = gid.x;
  var sy = gid.y;
  let w = params.width;
  let h = params.height;
  if (params.mode == 0u) {
    // Horizontal mirror
    if (gid.x >= w / 2u) { sx = w - 1u - gid.x; }
  } else if (params.mode == 1u) {
    // Vertical mirror
    if (gid.y >= h / 2u) { sy = h - 1u - gid.y; }
  } else {
    // Angular kaleidoscope
    let cx = f32(w) / 2.0;
    let cy = f32(h) / 2.0;
    let dx = f32(gid.x) - cx;
    let dy = f32(gid.y) - cy;
    let dist = sqrt(dx * dx + dy * dy);
    var angle = atan2(dy, dx) - params.angle;
    let seg_angle = 6.2831853 / f32(max(params.segments, 2u));
    angle = angle - floor(angle / seg_angle) * seg_angle;
    if (angle > seg_angle / 2.0) { angle = seg_angle - angle; }
    angle = angle + params.angle;
    sx = u32(clamp(cx + dist * cos(angle), 0.0, f32(w - 1u)));
    sy = u32(clamp(cy + dist * sin(angle), 0.0, f32(h - 1u)));
  }
  store_pixel(idx, load_pixel(sy * w + sx));
}
"#;

impl GpuFilter for MirrorKaleidoscope {
    fn shader_body(&self) -> &str { MIRROR_KALEIDOSCOPE_WGSL }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, self.segments);
        gpu_params_push_f32(&mut buf, self.angle);
        gpu_params_push_u32(&mut buf, self.mode);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}

// All effect filters are auto-registered via #[derive(V2Filter)] on their structs.

// ═══════════════════════════════════════════════════════════════════════════════
// GPU shaders for remaining CPU-only filters (loaded from .wgsl files)
// ═══════════════════════════════════════════════════════════════════════════════

impl GpuFilter for Emboss {
    fn shader_body(&self) -> &str {
        include_str!("../shaders/emboss.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, 0); // _pad0
        gpu_params_push_u32(&mut buf, 0); // _pad1
        buf
    }
}

const CHARCOAL_WGSL: &str = include_str!("../shaders/charcoal.wgsl");

impl GpuFilter for Charcoal {
    fn shader_body(&self) -> &str {
        CHARCOAL_WGSL
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}

const EFFECT_FILM_GRAIN_WGSL: &str = include_str!("../shaders/film_grain.wgsl");

impl GpuFilter for FilmGrain {
    fn shader_body(&self) -> &str {
        EFFECT_FILM_GRAIN_WGSL
    }
    fn workgroup_size(&self) -> [u32; 3] { [16, 16, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let inv_size = 1.0 / self.size.max(1.0);
        let seed = self.seed ^ noise::SEED_FILM_GRAIN;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, self.amount);
        gpu_params_push_f32(&mut buf, inv_size);
        gpu_params_push_u32(&mut buf, seed as u32);
        gpu_params_push_u32(&mut buf, (seed >> 32) as u32);
        gpu_params_push_u32(&mut buf, 0);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
    fn gpu_shader(&self, width: u32, height: u32) -> crate::node::GpuShader {
        crate::node::GpuShader {
            body: format!("{}\n{}", noise::NOISE_WGSL, EFFECT_FILM_GRAIN_WGSL),
            entry_point: "main",
            workgroup_size: self.workgroup_size(),
            params: self.params(width, height),
            extra_buffers: vec![],
            reduction_buffers: vec![],
            convergence_check: None,
            loop_dispatch: None,
        }
    }
}

impl GpuFilter for Pixelate {
    fn shader_body(&self) -> &str {
        include_str!("../shaders/pixelate.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, self.block_size.max(1));
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}

impl GpuFilter for Halftone {
    fn shader_body(&self) -> &str {
        include_str!("../shaders/halftone.wgsl")
    }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let ds = self.dot_size.max(1.0);
        let freq = std::f32::consts::PI / ds;
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_f32(&mut buf, freq);
        gpu_params_push_f32(&mut buf, (15.0 + self.angle_offset).to_radians()); // C angle
        gpu_params_push_f32(&mut buf, (75.0 + self.angle_offset).to_radians()); // M angle
        gpu_params_push_f32(&mut buf, (0.0 + self.angle_offset).to_radians());  // Y angle
        gpu_params_push_f32(&mut buf, (45.0 + self.angle_offset).to_radians()); // K angle
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}

impl GpuFilter for OilPaint {
    fn shader_body(&self) -> &str {
        include_str!("../shaders/oil_paint.wgsl")
    }
    fn workgroup_size(&self) -> [u32; 3] { [8, 8, 1] }
    fn params(&self, width: u32, height: u32) -> Vec<u8> {
        let mut buf = gpu_params_wh(width, height);
        gpu_params_push_u32(&mut buf, self.radius);
        gpu_params_push_u32(&mut buf, 0);
        buf
    }
}

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

    // ─── Noise filters ──────────────────────────────────────────────────

    #[test]
    fn gaussian_noise_deterministic() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let n = GaussianNoise { amount: 50.0, mean: 0.0, sigma: 25.0, seed: 42 };
        let a = n.compute(&input, 8, 8).unwrap();
        let b = n.compute(&input, 8, 8).unwrap();
        assert_eq!(a, b); // same seed → same output
    }

    #[test]
    fn gaussian_noise_zero_amount_identity() {
        let input = gradient_rgba(8, 8);
        let n = GaussianNoise { amount: 0.0, mean: 0.0, sigma: 25.0, seed: 42 };
        assert_eq!(n.compute(&input, 8, 8).unwrap(), input);
    }

    #[test]
    fn uniform_noise_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
        let n = UniformNoise { range: 50.0, seed: 42 };
        let out = n.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn salt_pepper_modifies_some_pixels() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let n = SaltPepperNoise { density: 0.5, seed: 42 };
        let out = n.compute(&input, 16, 16).unwrap();
        let changed = out.chunks_exact(4).zip(input.chunks_exact(4))
            .filter(|(a, b)| (a[0] - b[0]).abs() > 0.01)
            .count();
        assert!(changed > 0); // some pixels changed
        assert!(changed < 256); // not all
    }

    #[test]
    fn poisson_noise_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.3]);
        let n = PoissonNoise { scale: 50.0, seed: 42 };
        let out = n.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn film_grain_preserves_alpha() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 0.7]);
        let fg = FilmGrain { amount: 0.3, size: 2.0, seed: 42 };
        let out = fg.compute(&input, 16, 16).unwrap();
        assert!((out[3] - 0.7).abs() < 1e-6);
    }

    // ─── Pixelate ───────────────────────────────────────────────────────

    #[test]
    fn pixelate_uniform_block() {
        let input = gradient_rgba(16, 16);
        let p = Pixelate { block_size: 4 };
        let out = p.compute(&input, 16, 16).unwrap();
        // All pixels in first block should be identical
        let first = &out[0..4];
        let second = &out[4..8]; // next pixel in same block
        assert_eq!(first, second);
    }

    #[test]
    fn pixelate_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.3]);
        let p = Pixelate { block_size: 4 };
        let out = p.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.3).abs() < 1e-6);
    }

    // ─── Halftone ───────────────────────────────────────────────────────

    #[test]
    fn halftone_runs() {
        let input = gradient_rgba(32, 32);
        let ht = Halftone { dot_size: 4.0, angle_offset: 0.0 };
        let out = ht.compute(&input, 32, 32).unwrap();
        assert_eq!(out.len(), input.len());
    }

    // ─── Oil Paint ──────────────────────────────────────────────────────

    #[test]
    fn oil_paint_solid_unchanged() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let op = OilPaint { radius: 2 };
        let out = op.compute(&input, 8, 8).unwrap();
        assert!((out[0] - 0.5).abs() < 0.01);
    }

    // ─── Emboss ─────────────────────────────────────────────────────────

    #[test]
    fn emboss_solid_produces_consistent_output() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        let e = Emboss;
        let out = e.compute(&input, 8, 8).unwrap();
        // Solid → emboss kernel sum=1 → 0.5*1 + 0.5 offset = 1.0 for interior
        // All interior pixels should be the same value
        let center = (4 * 8 + 4) * 4;
        let neighbor = (4 * 8 + 5) * 4;
        assert!((out[center] - out[neighbor]).abs() < 1e-5);
    }

    #[test]
    fn emboss_preserves_alpha() {
        let input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 0.7]);
        let out = Emboss.compute(&input, 8, 8).unwrap();
        assert!((out[3] - 0.7).abs() < 1e-6);
    }

    // ─── Charcoal ───────────────────────────────────────────────────────

    #[test]
    fn charcoal_solid_white() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let c = Charcoal { radius: 1.0, sigma: 1.0 };
        let out = c.compute(&input, 16, 16).unwrap();
        // Solid → no edges → inverted = white
        let center = (8 * 16 + 8) * 4;
        assert!(out[center] > 0.9);
    }

    // ─── Chromatic effects ──────────────────────────────────────────────

    #[test]
    fn chromatic_split_zero_offset_identity() {
        let input = gradient_rgba(8, 8);
        let cs = ChromaticSplit {
            red_dx: 0.0, red_dy: 0.0,
            green_dx: 0.0, green_dy: 0.0,
            blue_dx: 0.0, blue_dy: 0.0,
        };
        let out = cs.compute(&input, 8, 8).unwrap();
        for (a, b) in input.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn chromatic_aberration_center_unchanged() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let ca = ChromaticAberration { strength: 5.0 };
        let out = ca.compute(&input, 16, 16).unwrap();
        // Solid color → no visible aberration
        assert!((out[0] - 0.5).abs() < 0.01);
    }

    // ─── Glitch ─────────────────────────────────────────────────────────

    #[test]
    fn glitch_deterministic() {
        let input = gradient_rgba(16, 16);
        let g = Glitch { shift_amount: 10.0, channel_offset: 3.0, intensity: 0.5, band_height: 4, seed: 42 };
        let a = g.compute(&input, 16, 16).unwrap();
        let b = g.compute(&input, 16, 16).unwrap();
        assert_eq!(a, b);
    }

    // ─── Light Leak ─────────────────────────────────────────────────────

    #[test]
    fn light_leak_brightens_center() {
        let input = solid_rgba(16, 16, [0.3, 0.3, 0.3, 1.0]);
        let ll = LightLeak {
            intensity: 0.8, position_x: 0.5, position_y: 0.5,
            radius: 0.5, warmth: 0.8,
        };
        let out = ll.compute(&input, 16, 16).unwrap();
        let center = (8 * 16 + 8) * 4;
        assert!(out[center] > 0.3); // center should be brighter
    }

    // ─── Mirror Kaleidoscope ────────────────────────────────────────────

    #[test]
    fn mirror_horizontal_symmetry() {
        let input = gradient_rgba(16, 16);
        let mk = MirrorKaleidoscope { segments: 2, angle: 0.0, mode: 0 };
        let out = mk.compute(&input, 16, 16).unwrap();
        assert_eq!(out.len(), input.len());
    }

    // ─── Output sizes ───────────────────────────────────────────────────

    #[test]
    fn all_output_sizes_correct() {
        let input = gradient_rgba(16, 16);
        let n = 16 * 16 * 4;

        assert_eq!(GaussianNoise { amount: 10.0, mean: 0.0, sigma: 25.0, seed: 42 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(UniformNoise { range: 50.0, seed: 42 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(SaltPepperNoise { density: 0.1, seed: 42 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(PoissonNoise { scale: 50.0, seed: 42 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(FilmGrain { amount: 0.3, size: 2.0, seed: 42 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(Pixelate { block_size: 4 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(Halftone { dot_size: 4.0, angle_offset: 0.0 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(OilPaint { radius: 2 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(Emboss.compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(Charcoal { radius: 1.0, sigma: 1.0 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(Glitch { shift_amount: 10.0, channel_offset: 3.0, intensity: 0.5, band_height: 4, seed: 42 }
            .compute(&input, 16, 16).unwrap().len(), n);
        assert_eq!(LightLeak { intensity: 0.5, position_x: 0.5, position_y: 0.5, radius: 0.5, warmth: 0.5 }
            .compute(&input, 16, 16).unwrap().len(), n);
    }

    // ─── HDR values ─────────────────────────────────────────────────────

    #[test]
    fn hdr_values_not_clamped() {
        let input = solid_rgba(4, 4, [5.0, -0.5, 100.0, 1.0]);
        let n = GaussianNoise { amount: 10.0, mean: 0.0, sigma: 25.0, seed: 42 };
        let out = n.compute(&input, 4, 4).unwrap();
        // HDR values should survive (not clamped to [0,1])
        assert!(out.chunks_exact(4).any(|p| p[0] > 1.0));
    }

    #[test]
    fn gpu_shaders_are_valid_wgsl_structure() {
        // Just verify all GPU shader bodies contain required WGSL elements
        let shaders: Vec<(&str, &str)> = vec![
            ("GaussianNoise", GAUSSIAN_NOISE_WGSL),
            ("UniformNoise", UNIFORM_NOISE_WGSL),
            ("SaltPepper", SALT_PEPPER_WGSL),
            ("PoissonNoise", POISSON_NOISE_WGSL),
            ("LightLeak", LIGHT_LEAK_WGSL),
            ("Glitch", GLITCH_WGSL),
            ("ChromaticSplit", CHROMATIC_SPLIT_WGSL),
            ("ChromaticAberration", CHROMATIC_ABERRATION_WGSL),
            ("MirrorKaleidoscope", MIRROR_KALEIDOSCOPE_WGSL),
            ("Emboss", include_str!("../shaders/emboss.wgsl")),
            ("Charcoal", CHARCOAL_WGSL),
            ("FilmGrain", EFFECT_FILM_GRAIN_WGSL),
            ("Pixelate", include_str!("../shaders/pixelate.wgsl")),
            ("Halftone", include_str!("../shaders/halftone.wgsl")),
            ("OilPaint", include_str!("../shaders/oil_paint.wgsl")),
        ];
        for (name, body) in shaders {
            assert!(body.contains("@compute"), "{name} missing @compute");
            assert!(body.contains("fn main("), "{name} missing fn main");
            assert!(body.contains("struct Params"), "{name} missing Params struct");
            assert!(
                body.contains("load_pixel") || body.contains("store_pixel"),
                "{name} missing pixel I/O"
            );
        }
    }

    #[test]
    fn gpu_noise_params_sizes_correct() {
        let g = GaussianNoise { amount: 50.0, mean: 0.0, sigma: 25.0, seed: 42 };
        assert_eq!(g.params(100, 100).len() % 4, 0, "GaussianNoise params not 4-byte aligned");
        let u = UniformNoise { range: 50.0, seed: 42 };
        assert_eq!(u.params(100, 100).len() % 4, 0, "UniformNoise params not 4-byte aligned");
        let sp = SaltPepperNoise { density: 0.05, seed: 42 };
        assert_eq!(sp.params(100, 100).len() % 4, 0, "SaltPepper params not 4-byte aligned");
        let p = PoissonNoise { scale: 100.0, seed: 42 };
        assert_eq!(p.params(100, 100).len() % 4, 0, "Poisson params not 4-byte aligned");
        // New GPU filters
        let e = Emboss;
        assert_eq!(e.params(100, 100).len() % 4, 0, "Emboss params not 4-byte aligned");
        let ch = Charcoal { radius: 1.0, sigma: 1.0 };
        assert_eq!(ch.params(100, 100).len() % 4, 0, "Charcoal params not 4-byte aligned");
        let fg = FilmGrain { amount: 0.3, size: 2.0, seed: 42 };
        assert_eq!(fg.params(100, 100).len() % 4, 0, "FilmGrain params not 4-byte aligned");
        let px = Pixelate { block_size: 4 };
        assert_eq!(px.params(100, 100).len() % 4, 0, "Pixelate params not 4-byte aligned");
        let ht = Halftone { dot_size: 4.0, angle_offset: 0.0 };
        assert_eq!(ht.params(100, 100).len() % 4, 0, "Halftone params not 4-byte aligned");
        let op = OilPaint { radius: 3 };
        assert_eq!(op.params(100, 100).len() % 4, 0, "OilPaint params not 4-byte aligned");
    }
}
