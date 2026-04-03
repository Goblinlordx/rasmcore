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
use crate::ops::Filter;

// ─── PRNG helpers ───────────────────────────────────────────────────────────

/// Simple deterministic PRNG (SplitMix64). Good enough for noise filters.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Gaussian via Box-Muller (pair generation, return one).
    #[inline]
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Deterministic hash noise in [-1.0, 1.0].
#[inline]
fn hash_noise(x: u32, seed: u32) -> f32 {
    let mut h = x
        .wrapping_mul(374761393)
        .wrapping_add(seed.wrapping_mul(1274126177));
    h = (h ^ (h >> 13)).wrapping_mul(1103515245);
    h = h ^ (h >> 16);
    (h as f32 / u32::MAX as f32) * 2.0 - 1.0
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

/// Luminance (Rec. 709).
#[inline]
fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

// ═══════════════════════════════════════════════════════════════════════════════
// Noise filters
// ═══════════════════════════════════════════════════════════════════════════════

/// Gaussian noise — additive normally-distributed noise.
#[derive(Clone)]
pub struct GaussianNoise {
    pub amount: f32,
    pub mean: f32,
    pub sigma: f32,
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
        let mut rng = Rng::new(self.seed);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            for ch in &mut pixel[..3] {
                let noise = mean + sigma * rng.next_gaussian();
                *ch += noise * amount;
            }
        }
        Ok(out)
    }
}

/// Uniform noise — additive uniformly-distributed noise.
#[derive(Clone)]
pub struct UniformNoise {
    pub range: f32,
    pub seed: u64,
}

impl Filter for UniformNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.range <= 0.0 {
            return Ok(input.to_vec());
        }
        let range = self.range / 255.0;
        let mut rng = Rng::new(self.seed);
        let mut out = input.to_vec();

        for pixel in out.chunks_exact_mut(4) {
            for ch in &mut pixel[..3] {
                let noise = (rng.next_f32() * 2.0 - 1.0) * range;
                *ch += noise;
            }
        }
        Ok(out)
    }
}

/// Salt-and-pepper noise — randomly replace pixels with black or white.
#[derive(Clone)]
pub struct SaltPepperNoise {
    pub density: f32,
    pub seed: u64,
}

impl Filter for SaltPepperNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        let mut rng = Rng::new(self.seed);
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
#[derive(Clone)]
pub struct PoissonNoise {
    pub scale: f32,
    pub seed: u64,
}

impl Filter for PoissonNoise {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let _ = (width, height);
        if self.scale <= 0.0 {
            return Ok(input.to_vec());
        }
        let mut rng = Rng::new(self.seed);
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
#[derive(Clone)]
pub struct FilmGrain {
    pub amount: f32,
    pub size: f32,
    pub seed: u64,
}

impl Filter for FilmGrain {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.amount <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let mut rng = Rng::new(self.seed);

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
#[derive(Clone)]
pub struct Pixelate {
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
#[derive(Clone)]
pub struct Halftone {
    pub dot_size: f32,
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
#[derive(Clone)]
pub struct OilPaint {
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
#[derive(Clone)]
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
#[derive(Clone)]
pub struct Charcoal {
    pub radius: f32,
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
#[derive(Clone)]
pub struct ChromaticSplit {
    pub red_dx: f32,
    pub red_dy: f32,
    pub green_dx: f32,
    pub green_dy: f32,
    pub blue_dx: f32,
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
#[derive(Clone)]
pub struct ChromaticAberration {
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
#[derive(Clone)]
pub struct Glitch {
    pub shift_amount: f32,
    pub channel_offset: f32,
    pub intensity: f32,
    pub band_height: u32,
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
            let noise = hash_noise(band as u32, self.seed);

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
#[derive(Clone)]
pub struct LightLeak {
    pub intensity: f32,
    pub position_x: f32,
    pub position_y: f32,
    pub radius: f32,
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
#[derive(Clone)]
pub struct MirrorKaleidoscope {
    pub segments: u32,
    pub angle: f32,
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
}
