//! Spatial filters — neighborhood operations on f32 pixel data.
//!
//! All operate on `&[f32]` RGBA (4 channels). Input includes overlap region
//! (expanded by the pipeline via `SpatialFilter::overlap_radius`). Output
//! matches input dimensions. The FilterNode wrapper handles cropping to the
//! requested tile.
//!
//! No format dispatch. No u8/u16 paths. Just f32.

use crate::node::PipelineError;
use crate::ops::Filter;

// ─── Helpers ──────────────────────────────────────────────────────────────────

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

/// Generate a 1D Gaussian kernel (normalized to sum=1).
fn gaussian_kernel_1d(radius: f32) -> Vec<f32> {
    let sigma = radius;
    let ksize = ((sigma * 6.0 + 1.0).round() as usize) | 1; // ensure odd
    let ksize = ksize.max(3);
    let center = ksize / 2;
    let mut kernel = Vec::with_capacity(ksize);
    let mut sum = 0.0f32;
    for i in 0..ksize {
        let x = i as f32 - center as f32;
        let w = (-0.5 * (x / sigma).powi(2)).exp();
        kernel.push(w);
        sum += w;
    }
    let inv = 1.0 / sum;
    for w in &mut kernel {
        *w *= inv;
    }
    kernel
}

// ─── Gaussian Blur ────────────────────────────────────────────────────────────

/// Gaussian blur — separable convolution on f32 data.
#[derive(Clone)]
pub struct GaussianBlur {
    pub radius: f32,
}

impl Filter for GaussianBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let kernel = gaussian_kernel_1d(self.radius);
        let r = kernel.len() / 2;

        // Pass 1: horizontal
        let mut tmp = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for kx in 0..kernel.len() {
                    let sx = clamp_coord(x as i32 + kx as i32 - r as i32, w);
                    let idx = (y * w + sx) * 4;
                    for c in 0..4 {
                        sum[c] += kernel[kx] * input[idx + c];
                    }
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    tmp[out_idx + c] = sum[c];
                }
            }
        }

        // Pass 2: vertical
        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for ky in 0..kernel.len() {
                    let sy = clamp_coord(y as i32 + ky as i32 - r as i32, h);
                    let idx = (sy * w + x) * 4;
                    for c in 0..4 {
                        sum[c] += kernel[ky] * tmp[idx + c];
                    }
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[out_idx + c] = sum[c];
                }
            }
        }

        Ok(out)
    }
}

impl GaussianBlur {
    pub fn overlap_radius(&self) -> u32 {
        (self.radius * 3.0).ceil() as u32
    }
}

// ─── Box Blur ─────────────────────────────────────────────────────────────────

/// Box blur — running average within radius. O(1) per pixel.
#[derive(Clone)]
pub struct BoxBlur {
    pub radius: u32,
}

impl Filter for BoxBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as i32;
        let diameter = (2 * r + 1) as f32;
        let inv_d = 1.0 / diameter;

        // Horizontal pass
        let mut tmp = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for dx in -r..=r {
                    let sx = clamp_coord(x as i32 + dx, w);
                    let idx = (y * w + sx) * 4;
                    for c in 0..4 {
                        sum[c] += input[idx + c];
                    }
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    tmp[out_idx + c] = sum[c] * inv_d;
                }
            }
        }

        // Vertical pass
        let mut out = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for dy in -r..=r {
                    let sy = clamp_coord(y as i32 + dy, h);
                    let idx = (sy * w + x) * 4;
                    for c in 0..4 {
                        sum[c] += tmp[idx + c];
                    }
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[out_idx + c] = sum[c] * inv_d;
                }
            }
        }

        Ok(out)
    }
}

// ─── Sharpen (Unsharp Mask) ───────────────────────────────────────────────────

/// Unsharp mask sharpening — enhances edges by subtracting blurred from original.
///
/// `output = input + amount * (input - blur(input, radius))`
#[derive(Clone)]
pub struct Sharpen {
    pub radius: f32,
    pub amount: f32,
}

impl Filter for Sharpen {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.radius };
        let blurred = blur.compute(input, width, height)?;
        let amount = self.amount;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            // Alpha (every 4th channel) preserved unchanged
            if i % 4 == 3 {
                out.push(v);
            } else {
                out.push(v + amount * (v - blurred[i]));
            }
        }
        Ok(out)
    }
}

// ─── Median ───────────────────────────────────────────────────────────────────

/// Median filter — replaces each pixel with median of its neighborhood.
///
/// Effective for salt-and-pepper noise removal while preserving edges.
#[derive(Clone)]
pub struct Median {
    pub radius: u32,
}

impl Filter for Median {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.radius == 0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let r = self.radius as i32;
        let mut out = vec![0.0f32; w * h * 4];
        let mut neighborhood = Vec::new();

        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    if c == 3 {
                        // Alpha: pass through
                        out[(y * w + x) * 4 + 3] = input[(y * w + x) * 4 + 3];
                        continue;
                    }
                    neighborhood.clear();
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let sx = clamp_coord(x as i32 + dx, w);
                            let sy = clamp_coord(y as i32 + dy, h);
                            neighborhood.push(input[(sy * w + sx) * 4 + c]);
                        }
                    }
                    neighborhood.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    out[(y * w + x) * 4 + c] = neighborhood[neighborhood.len() / 2];
                }
            }
        }

        Ok(out)
    }
}

// ─── General Convolution ──────────────────────────────────────────────────────

/// General NxN convolution with arbitrary kernel.
#[derive(Clone)]
pub struct Convolve {
    pub kernel: Vec<f32>,
    pub kernel_width: u32,
    pub kernel_height: u32,
    pub divisor: f32,
}

impl Filter for Convolve {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let kw = self.kernel_width as usize;
        let kh = self.kernel_height as usize;
        let rw = kw / 2;
        let rh = kh / 2;
        let inv_div = 1.0 / self.divisor;
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for ky in 0..kh {
                    for kx in 0..kw {
                        let sx = clamp_coord(x as i32 + kx as i32 - rw as i32, w);
                        let sy = clamp_coord(y as i32 + ky as i32 - rh as i32, h);
                        let k = self.kernel[ky * kw + kx];
                        let idx = (sy * w + sx) * 4;
                        for c in 0..4 {
                            sum[c] += k * input[idx + c];
                        }
                    }
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[out_idx + c] = sum[c] * inv_div;
                }
            }
        }

        Ok(out)
    }
}

// ─── Bilateral Filter ─────────────────────────────────────────────────────────

/// Bilateral filter — edge-preserving smoothing.
///
/// Weights pixels by both spatial distance and color similarity.
#[derive(Clone)]
pub struct Bilateral {
    pub diameter: u32,
    pub sigma_color: f32,
    pub sigma_space: f32,
}

impl Filter for Bilateral {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let r = (self.diameter / 2) as i32;
        let sc2 = -0.5 / (self.sigma_color * self.sigma_color);
        let ss2 = -0.5 / (self.sigma_space * self.sigma_space);
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let center_idx = (y * w + x) * 4;
                let mut sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = clamp_coord(x as i32 + dx, w);
                        let sy = clamp_coord(y as i32 + dy, h);
                        let idx = (sy * w + sx) * 4;

                        // Spatial weight
                        let dist2 = (dx * dx + dy * dy) as f32;
                        let ws = (dist2 * ss2).exp();

                        // Color weight (Euclidean distance in RGB)
                        let dr = input[idx] - input[center_idx];
                        let dg = input[idx + 1] - input[center_idx + 1];
                        let db = input[idx + 2] - input[center_idx + 2];
                        let color_dist2 = dr * dr + dg * dg + db * db;
                        let wc = (color_dist2 * sc2).exp();

                        let weight = ws * wc;
                        for c in 0..3 {
                            sum[c] += weight * input[idx + c];
                        }
                        weight_sum += weight;
                    }
                }

                let inv_w = if weight_sum > 1e-10 { 1.0 / weight_sum } else { 0.0 };
                for c in 0..3 {
                    out[center_idx + c] = sum[c] * inv_w;
                }
                out[center_idx + 3] = input[center_idx + 3]; // alpha
            }
        }

        Ok(out)
    }
}

// ─── Motion Blur ──────────────────────────────────────────────────────────────

/// Motion blur — linear directional blur.
#[derive(Clone)]
pub struct MotionBlur {
    pub angle: f32,  // degrees
    pub length: f32, // pixels
}

impl Filter for MotionBlur {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        if self.length <= 0.0 {
            return Ok(input.to_vec());
        }
        let w = width as usize;
        let h = height as usize;
        let rad = self.angle.to_radians();
        let dx = rad.cos();
        let dy = rad.sin();
        let steps = self.length.ceil() as usize;
        let inv_steps = 1.0 / (steps as f32 + 1.0);
        let mut out = vec![0.0f32; w * h * 4];

        for y in 0..h {
            for x in 0..w {
                let mut sum = [0.0f32; 4];
                for s in 0..=steps {
                    let t = s as f32 - steps as f32 * 0.5;
                    let sx = clamp_coord((x as f32 + t * dx).round() as i32, w);
                    let sy = clamp_coord((y as f32 + t * dy).round() as i32, h);
                    let idx = (sy * w + sx) * 4;
                    for c in 0..4 {
                        sum[c] += input[idx + c];
                    }
                }
                let out_idx = (y * w + x) * 4;
                for c in 0..4 {
                    out[out_idx + c] = sum[c] * inv_steps;
                }
            }
        }

        Ok(out)
    }
}

// ─── High Pass ────────────────────────────────────────────────────────────────

/// High pass filter — subtracts blur from original, adding mid-gray offset.
///
/// `output = (input - blur(input)) + 0.5`
#[derive(Clone)]
pub struct HighPass {
    pub radius: f32,
}

impl Filter for HighPass {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let blur = GaussianBlur { radius: self.radius };
        let blurred = blur.compute(input, width, height)?;
        let mut out = Vec::with_capacity(input.len());
        for (i, &v) in input.iter().enumerate() {
            if i % 4 == 3 {
                out.push(v); // alpha
            } else {
                out.push((v - blurred[i]) + 0.5);
            }
        }
        Ok(out)
    }
}

// ─── Displacement Map ─────────────────────────────────────────────────────────

/// Displacement map — warp pixels by per-pixel offset fields.
///
/// `map_x` and `map_y` are f32 slices of absolute source coordinates.
#[derive(Clone)]
pub struct DisplacementMap {
    pub map_x: Vec<f32>,
    pub map_y: Vec<f32>,
}

impl Filter for DisplacementMap {
    fn compute(&self, input: &[f32], width: u32, height: u32) -> Result<Vec<f32>, PipelineError> {
        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        if self.map_x.len() != n || self.map_y.len() != n {
            return Err(PipelineError::InvalidParams(format!(
                "displacement map size mismatch: expected {n}, got x={} y={}",
                self.map_x.len(),
                self.map_y.len()
            )));
        }

        let mut out = vec![0.0f32; n * 4];
        let wi = w as i32;
        let hi = h as i32;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let sx = self.map_x[idx];
                let sy = self.map_y[idx];

                // Bilinear interpolation
                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                let sample = |px: i32, py: i32, c: usize| -> f32 {
                    if px >= 0 && px < wi && py >= 0 && py < hi {
                        input[(py as usize * w + px as usize) * 4 + c]
                    } else {
                        0.0
                    }
                };

                let out_off = idx * 4;
                for c in 0..4 {
                    let v = sample(x0, y0, c) * (1.0 - fx) * (1.0 - fy)
                        + sample(x0 + 1, y0, c) * fx * (1.0 - fy)
                        + sample(x0, y0 + 1, c) * (1.0 - fx) * fy
                        + sample(x0 + 1, y0 + 1, c) * fx * fy;
                    out[out_off + c] = v;
                }
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, color: [f32; 4]) -> Vec<f32> {
        let n = w as usize * h as usize;
        let mut px = Vec::with_capacity(n * 4);
        for _ in 0..n {
            px.extend_from_slice(&color);
        }
        px
    }

    fn gradient_rgba(w: u32, h: u32) -> Vec<f32> {
        let mut px = Vec::with_capacity(w as usize * h as usize * 4);
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

    #[test]
    fn gaussian_blur_solid_unchanged() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let blur = GaussianBlur { radius: 3.0 };
        let output = blur.compute(&input, 16, 16).unwrap();
        // Solid color blurred = same color
        assert!((output[0] - 0.5).abs() < 0.01);
        assert!((output[1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn gaussian_blur_zero_radius_identity() {
        let input = gradient_rgba(8, 8);
        let blur = GaussianBlur { radius: 0.0 };
        let output = blur.compute(&input, 8, 8).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn box_blur_solid_unchanged() {
        let input = solid_rgba(16, 16, [0.3, 0.6, 0.9, 1.0]);
        let blur = BoxBlur { radius: 2 };
        let output = blur.compute(&input, 16, 16).unwrap();
        assert!((output[0] - 0.3).abs() < 0.01);
        assert!((output[1] - 0.6).abs() < 0.01);
    }

    #[test]
    fn sharpen_preserves_solid() {
        let input = solid_rgba(16, 16, [0.5, 0.5, 0.5, 1.0]);
        let sharp = Sharpen { radius: 2.0, amount: 1.0 };
        let output = sharp.compute(&input, 16, 16).unwrap();
        // Sharpening a solid image = no change (no edges)
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn median_removes_outlier() {
        let mut input = solid_rgba(8, 8, [0.5, 0.5, 0.5, 1.0]);
        // Set center pixel to an outlier
        let center = (4 * 8 + 4) * 4;
        input[center] = 1.0;
        input[center + 1] = 1.0;
        input[center + 2] = 1.0;

        let med = Median { radius: 1 };
        let output = med.compute(&input, 8, 8).unwrap();
        // Median should replace outlier with neighborhood median (~0.5)
        assert!((output[center] - 0.5).abs() < 0.01);
    }

    #[test]
    fn convolve_identity_kernel() {
        let input = gradient_rgba(8, 8);
        let identity = Convolve {
            kernel: vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            kernel_width: 3,
            kernel_height: 3,
            divisor: 1.0,
        };
        let output = identity.compute(&input, 8, 8).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn bilateral_preserves_solid() {
        let input = solid_rgba(8, 8, [0.4, 0.6, 0.8, 1.0]);
        let bilat = Bilateral {
            diameter: 5,
            sigma_color: 0.1,
            sigma_space: 1.0,
        };
        let output = bilat.compute(&input, 8, 8).unwrap();
        assert!((output[0] - 0.4).abs() < 0.01);
        assert!((output[1] - 0.6).abs() < 0.01);
    }

    #[test]
    fn motion_blur_zero_length_identity() {
        let input = gradient_rgba(8, 8);
        let mb = MotionBlur { angle: 0.0, length: 0.0 };
        let output = mb.compute(&input, 8, 8).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn high_pass_solid_produces_midgray() {
        let input = solid_rgba(16, 16, [0.3, 0.3, 0.3, 1.0]);
        let hp = HighPass { radius: 3.0 };
        let output = hp.compute(&input, 16, 16).unwrap();
        // Solid → blur = same → high_pass = 0 + 0.5 = 0.5
        assert!((output[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn hdr_values_survive_blur() {
        let input = solid_rgba(8, 8, [5.0, -0.5, 100.0, 1.0]);
        let blur = GaussianBlur { radius: 2.0 };
        let output = blur.compute(&input, 8, 8).unwrap();
        // HDR values not clamped
        assert!((output[0] - 5.0).abs() < 0.1);
        assert!((output[2] - 100.0).abs() < 1.0);
    }

    #[test]
    fn displacement_map_identity() {
        let (w, h) = (8u32, 8u32);
        let input = gradient_rgba(w, h);
        let mut map_x = Vec::with_capacity((w * h) as usize);
        let mut map_y = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                map_x.push(x as f32);
                map_y.push(y as f32);
            }
        }
        let dm = DisplacementMap { map_x, map_y };
        let output = dm.compute(&input, w, h).unwrap();
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn output_sizes_correct() {
        let input = gradient_rgba(32, 32);
        let n = 32 * 32 * 4;

        assert_eq!(GaussianBlur { radius: 3.0 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(BoxBlur { radius: 2 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(Sharpen { radius: 2.0, amount: 1.0 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(Median { radius: 1 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(HighPass { radius: 2.0 }.compute(&input, 32, 32).unwrap().len(), n);
        assert_eq!(MotionBlur { angle: 45.0, length: 5.0 }.compute(&input, 32, 32).unwrap().len(), n);
    }
}
